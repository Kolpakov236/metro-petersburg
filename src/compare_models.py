"""Сравнение моделей (доп. эксперимент): справились бы другие модели лучше замороженной LightGBM на тех же условиях?

Запуск:
  python -m src.compare_models               # май–август: все модели, выбор → reports/compare/
  python -m src.compare_models --reuse       # то же, сырые прогнозы — из кэша data/interim/compare_raw/
  python -m src.compare_models --september   # один прогон на сентябре «для информации» (после выбора)

Те же 4 фолда (BT.folds, BT.split), строки, признаки финальной модели, цель z = log((y+1)/(f_ref+1)), веса f_ref,
окна (C — последние 28 суток обучения, E — последние 14) и та же асимметричная CQR по ячейкам «период суток × h»,
что у замороженной `model.LGBMQuantile`. Меняется только «ученик» — модель, которая учит z:
  - LightGBM — эталон: внутри этой схемы обязан совпасть с замороженными прогнозами (проверяется при запуске);
  - линейная квантильная регрессия — statsmodels QuantReg; веса — через однородность pinball:
    w·ρ(y − xβ) = ρ(w·y − w·xβ), то есть обычная QuantReg на (w·z, w·X);
  - CatBoost — MultiQuantile, одна модель на три квантиля;
  - XGBoost — reg:quantileerror, три модели;
  - ансамбль LightGBM + CatBoost — сырые z учеников усредняются до обратного перевода с весом w, выбранным на окне C
    внутри обучения фолда (минимум WAPE q50 отложенных моделей), затем та же поправка.
Гиперпараметры — умеренные по умолчанию, без подбора. Замороженные модули только импортируются. LightGBM берётся
из кэша сырых прогнозов (data/interim/compare_raw/), если он есть, и не переобучается.
Официальной остаётся замороженная LightGBM. Кандидат во «вторую версию» — модель с ΔWAPE к ней > 0, 95 % ДИ выше нуля
и ΔWAPE > 0 хотя бы в 3 фолдах из 4 (май–август).
"""
import argparse
import hashlib
import json
import pickle
import time
import warnings
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

from src import backtest as BT
from src import config
from src import features as F
from src import model as M

OUT = config.ROOT / "reports" / "compare"
RAW = config.INTERIM / "compare_raw"
CACHE_VERSION = 1
QUANTILES = M.QUANTILES
SEED, THREADS = 0, 8
MAX_ROUNDS, ES_ROUNDS = M.MAX_ROUNDS, M.ES_ROUNDS
PRED_COLS = ["vestibule_id", "station_id", "group", "h", "tau", "hour", "band", "sday", "y", "is_special",
             "is_anomaly", "is_holiday"]
LABELS = {"lgbm": "LightGBM (замороженная)", "linear": "Линейная квантильная регрессия", "catboost": "CatBoost",
          "xgboost": "XGBoost", "lgbm_catboost": "LightGBM + CatBoost (w по окну C)"}
PAIRS = {"lgbm_catboost": ("lgbm", "catboost")}                                  # вес w первой модели — по окну C
W_GRID = np.round(np.arange(0, 1.0001, 0.05), 2)
MIN_FOLDS = 3                                                                   # кандидат: ΔWAPE > 0 в ≥ 3 фолдах из 4


def target(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Цель z и веса — как у замороженной модели."""
    return M.to_z(df.y, df[M.F_REF], "ratio"), np.maximum(df[M.F_REF].to_numpy(float), 1.0)


# --- Ученики ----------------------------------------------------------------------
@dataclass
class LGBMLearner:
    """Три бустера LightGBM ровно как в `M.LGBMQuantile` (параметры, датасеты, ранняя остановка)."""
    features: list[str]
    params: dict = field(default_factory=lambda: dict(M.PARAMS))
    max_rounds: int = MAX_ROUNDS
    es_rounds: int = ES_ROUNDS
    name: str = "lgbm"
    iters: dict = field(default_factory=dict)
    boosters: dict = field(default_factory=dict)

    def _ds(self, df, reference=None):
        z, w = target(df)
        return lgb.Dataset(df[self.features], z, weight=w, reference=reference, free_raw_data=False)

    def _params(self, a):
        return {**self.params, "objective": "quantile", "alpha": a, "metric": "quantile"}

    def fit(self, df, valid=None):
        ds = self._ds(df)
        dv = self._ds(valid, reference=ds)
        for a in QUANTILES:
            b = lgb.train(self._params(a), ds, num_boost_round=self.max_rounds, valid_sets=[dv],
                          callbacks=[lgb.early_stopping(self.es_rounds, verbose=False)])
            self.iters[a] = max(int(b.best_iteration), 1)
            self.boosters[a] = b
        return self

    def refit(self, df):
        ds = self._ds(df)
        self.boosters = {a: lgb.train(self._params(a), ds, num_boost_round=self.iters[a]) for a in QUANTILES}
        return self

    def predict(self, df):
        return {a: b.predict(df[self.features], num_iteration=self.iters[a]) for a, b in self.boosters.items()}


@dataclass
class LinearLearner:
    """Линейная квантильная регрессия (statsmodels QuantReg), по модели на квантиль; ранней остановки нет.
    Подготовка признаков подгоняется только на обучающей части:
      - категории (вестибюль, тип дня, час, день недели) — one-hot без первой категории;
      - группа станций не берётся: её целиком задаёт вестибюль, колонки были бы линейно зависимы;
      - пропуски — медиана обучения плюс флаг пропуска (одинаковые флаги схлопываются);
      - числовые признаки стандартизуются."""
    features: list[str]
    max_iter: int = 2000
    name: str = "linear"
    iters: dict = field(default_factory=dict)
    prep: dict = field(default_factory=dict)
    beta: dict = field(default_factory=dict)

    CATS = ("vestibule", "day_type_tau", "hour_tau", "dow_tau")
    DROP = ("station_group",)

    def _fit_prep(self, df):
        cats = [c for c in self.CATS if c in self.features]
        num = [c for c in self.features if c not in cats and c not in self.DROP]
        X = df[num].astype(float)
        na = X.isna()
        flag_cols = [c for c in num if na[c].any()]
        flags = na[flag_cols].astype(float)
        keep = list(flags.T.drop_duplicates().index) if flag_cols else []
        levels = {c: sorted(df[c].astype(str).unique()) for c in cats}
        med = X.median()
        Xf = X.fillna(med)
        self.prep = {"cats": cats, "num": num, "flags": keep, "levels": levels, "median": med,
                     "mean": Xf.mean(), "std": Xf.std().replace(0, 1.0).fillna(1.0)}

    def _design(self, df) -> np.ndarray:
        p = self.prep
        X = df[p["num"]].astype(float)
        flags = X[p["flags"]].isna().to_numpy(float)
        Xs = ((X.fillna(p["median"]) - p["mean"]) / p["std"]).to_numpy()
        dummies = [(df[c].astype(str).to_numpy()[:, None] == np.array(lv[1:])[None, :]).astype(float)
                   for c, lv in p["levels"].items()]
        return np.column_stack([np.ones(len(df)), Xs, flags, *dummies])

    def fit(self, df, valid=None):
        import statsmodels.api as sm
        self._fit_prep(df)
        A = self._design(df)
        z, w = target(df)
        w = w / w.mean()                       # масштаб весов на решение не влияет, но обусловленность лучше
        for a in QUANTILES:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = sm.QuantReg(w * z, A * w[:, None]).fit(q=a, max_iter=self.max_iter)
            self.beta[a] = np.asarray(r.params)
            self.iters[a] = int(getattr(r, "iterations", 0) or 0)
        return self

    def refit(self, df):
        return self.fit(df)

    def predict(self, df):
        A = self._design(df)
        return {a: A @ b for a, b in self.beta.items()}


@dataclass
class CatBoostLearner:
    """CatBoost MultiQuantile: одна модель на три квантиля; ранняя остановка по E; категории нативно."""
    features: list[str]
    params: dict = field(default_factory=lambda: {
        "loss_function": "MultiQuantile:alpha=0.1,0.5,0.9", "learning_rate": 0.05, "depth": 6, "l2_leaf_reg": 3.0,
        "random_seed": SEED, "thread_count": THREADS, "verbose": False, "allow_writing_files": False})
    max_rounds: int = MAX_ROUNDS
    es_rounds: int = ES_ROUNDS
    name: str = "catboost"
    iters: dict = field(default_factory=dict)
    model: object = None

    def _pool(self, df):
        from catboost import Pool
        X = df[self.features].copy()
        cats = [c for c in self.features if c in F.CATEGORICAL]
        for c in cats:
            X[c] = X[c].astype(str)
        z, w = target(df)
        return Pool(X, z, weight=w, cat_features=cats)

    def fit(self, df, valid=None):
        from catboost import CatBoostRegressor
        self.model = CatBoostRegressor(**self.params, iterations=self.max_rounds, od_type="Iter",
                                       od_wait=self.es_rounds, use_best_model=True)
        self.model.fit(self._pool(df), eval_set=self._pool(valid))
        n = int(self.model.get_best_iteration()) + 1
        self.iters = {a: n for a in QUANTILES}
        return self

    def refit(self, df):
        from catboost import CatBoostRegressor
        self.model = CatBoostRegressor(**self.params, iterations=self.iters[QUANTILES[0]])
        self.model.fit(self._pool(df))
        return self

    def predict(self, df):
        p = np.asarray(self.model.predict(self._pool(df)))
        return {a: p[:, i] for i, a in enumerate(QUANTILES)}


@dataclass
class XGBLearner:
    """XGBoost reg:quantileerror, три модели; ранняя остановка по E; категории нативно (pandas category)."""
    features: list[str]
    params: dict = field(default_factory=lambda: {
        "objective": "reg:quantileerror", "eta": 0.05, "max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.9,
        "lambda": 1.0, "tree_method": "hist", "eval_metric": "quantile", "nthread": THREADS, "seed": SEED})
    max_rounds: int = MAX_ROUNDS
    es_rounds: int = ES_ROUNDS
    name: str = "xgboost"
    iters: dict = field(default_factory=dict)
    boosters: dict = field(default_factory=dict)

    def _dm(self, df):
        import xgboost as xgb
        z, w = target(df)
        return xgb.DMatrix(df[self.features], label=z, weight=w, enable_categorical=True)

    def fit(self, df, valid=None):
        import xgboost as xgb
        dt, dv = self._dm(df), self._dm(valid)
        for a in QUANTILES:
            b = xgb.train({**self.params, "quantile_alpha": a}, dt, self.max_rounds, evals=[(dv, "E")],
                          early_stopping_rounds=self.es_rounds, verbose_eval=False)
            self.iters[a] = int(b.best_iteration) + 1
            self.boosters[a] = b
        return self

    def refit(self, df):
        import xgboost as xgb
        dt = self._dm(df)
        self.boosters = {a: xgb.train({**self.params, "quantile_alpha": a}, dt, self.iters[a]) for a in QUANTILES}
        return self

    def predict(self, df):
        d = self._dm(df)
        return {a: b.predict(d, iteration_range=(0, self.iters[a])) for a, b in self.boosters.items()}


def learners(features: list[str]) -> dict:
    return {"lgbm": LGBMLearner(features), "linear": LinearLearner(features), "catboost": CatBoostLearner(features),
            "xgboost": XGBLearner(features)}


# --- Общая схема: окна, обучение, калибровка ---------------------------------------
@dataclass
class RawFold:
    """Сырые прогнозы z ученика в одном фолде: отложенной модели на окне C и дообученной — на тесте."""
    cal_index: np.ndarray
    z_cal: dict
    test_index: np.ndarray
    z_test: dict
    iters: dict
    seconds: float


def windows(sday: pd.Series, calib_days: int = 28, es_days: int = 14) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Начало окон C и E — как `M.LGBMQuantile.windows`."""
    last = sday.max()
    return last - pd.Timedelta(days=calib_days - 1), last - pd.Timedelta(days=es_days - 1)


def fit_raw(learner, train: pd.DataFrame, test: pd.DataFrame) -> RawFold:
    """Отложенный ученик на W \\ C с ранней остановкой по E → прогноз C; дообучение на W → прогноз теста."""
    c0, e0 = windows(train.sday)
    hold, cal, es = train[train.sday < c0], train[train.sday >= c0], train[train.sday >= e0]
    t = time.perf_counter()
    learner.fit(hold, valid=es)
    z_cal = learner.predict(cal)
    learner.refit(train)
    z_test = learner.predict(test)
    return RawFold(cal.index.to_numpy(), z_cal, test.index.to_numpy(), z_test, dict(learner.iters),
                   time.perf_counter() - t)


def combine(raws: list[RawFold], weights: list[float], info: dict | None = None) -> RawFold:
    """Ансамбль: взвешенная сумма сырых z учеников по каждому квантилю (до обратного перевода); время — сумма."""
    a = raws[0]
    assert all((r.cal_index == a.cal_index).all() and (r.test_index == a.test_index).all() for r in raws)
    mix = lambda zs: {q: sum(w * z[q] for w, z in zip(weights, zs)) for q in QUANTILES}
    return RawFold(a.cal_index, mix([r.z_cal for r in raws]), a.test_index, mix([r.z_test for r in raws]),
                   {**(info or {}), "weights": list(weights)}, sum(r.seconds for r in raws))


def average(a: RawFold, b: RawFold) -> RawFold:
    """Ансамбль двух учеников с равными весами."""
    return combine([a, b], [0.5, 0.5])


def choose_w(a: RawFold, b: RawFold, train: pd.DataFrame, grid=W_GRID) -> float:
    """Вес w модели a в паре: минимум WAPE q50 отложенных моделей на окне C (строки a.cal_index из обучения фолда).
    Тестовые строки не участвуют; при равенстве — меньший w."""
    cal = train.loc[a.cal_index]
    y, f = cal.y.to_numpy(float), cal[M.F_REF].to_numpy(float)
    err = [np.abs(M.from_z(w * a.z_cal[0.5] + (1 - w) * b.z_cal[0.5], f, "ratio") - y).sum() for w in grid]
    return float(grid[int(np.argmin(err))])


def ensembles(by_name: dict[str, RawFold], train: pd.DataFrame) -> dict[str, RawFold]:
    """Ансамбли фолда из готовых сырых прогнозов учеников."""
    out = {}
    for name, (x, y) in PAIRS.items():
        w = choose_w(by_name[x], by_name[y], train)
        out[name] = combine([by_name[x], by_name[y]], [w, 1 - w], {"w": w})
    return out


def calibrate(raw: RawFold, train: pd.DataFrame, test: pd.DataFrame, cells=("band", "h"),
              min_cell: int = 300) -> tuple[pd.DataFrame, dict]:
    """Асимметричная CQR по ошибкам на окне C и прогноз теста — те же формулы, что в `M.LGBMQuantile`."""
    cal = train.loc[raw.cal_index]
    zc = M.to_z(cal.y, cal[M.F_REF], "ratio")
    pc = raw.z_cal
    scores = pd.DataFrame({"lo": pc[0.1] - zc, "hi": zc - pc[0.9]}, index=cal.index)
    miss = QUANTILES[0]
    keys = [cal[c] for c in cells]
    sizes = scores.groupby(keys).size()
    corr = scores.groupby(keys).agg(lambda s: M.conformal(s.to_numpy(), miss))
    corr = corr[sizes >= min_cell]
    corr_all = scores.apply(lambda s: M.conformal(s.to_numpy(), miss))

    def corrections(rows):
        idx = pd.MultiIndex.from_frame(rows[list(cells)])
        return (corr.lo.reindex(idx).fillna(corr_all.lo).to_numpy(),
                corr.hi.reindex(idx).fillna(corr_all.hi).to_numpy())

    lo_c, hi_c = corrections(cal)
    rows = test.loc[raw.test_index]
    z = raw.z_test
    lo, hi = corrections(rows)
    f = rows[M.F_REF].to_numpy(float)
    inv = lambda v: M.from_z(v, f, "ratio")
    qs = np.sort(np.column_stack([inv(z[0.1] - lo), inv(z[0.5]), inv(z[0.9] + hi)]), axis=1)
    raw10, raw90 = np.minimum(inv(z[0.1]), qs[:, 1]), np.maximum(inv(z[0.9]), qs[:, 1])
    small = f < 1
    qs[small] = f[small, None]
    raw10[small], raw90[small] = f[small], f[small]
    p = pd.DataFrame({"q10": qs[:, 0], "q50": qs[:, 1], "q90": qs[:, 2], "q10_raw": raw10, "q90_raw": raw90},
                     index=rows.index)
    info = {"calib_cov_raw": float(np.mean((zc >= pc[0.1]) & (zc <= pc[0.9]))),
            "calib_cov": float(np.mean((zc >= pc[0.1] - lo_c) & (zc <= pc[0.9] + hi_c)))}
    return p, info


# --- Прогон по фолдам ---------------------------------------------------------------
def _key(name: str, learner, fold: BT.Fold, n_train: int) -> str:
    spec = {"v": CACHE_VERSION, "name": name, "features": learner.features, "fold": fold.name,
            "train_end": str(fold.train_end.date()), "n": n_train,
            "params": getattr(learner, "params", None), "rounds": getattr(learner, "max_rounds", None)}
    return hashlib.sha1(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()[:12]


def run_raw(rows: pd.DataFrame, fold_list: list[BT.Fold], features: list[str], reuse: bool = False,
            log=print) -> dict[str, dict[str, RawFold]]:
    """Сырые прогнозы всех учеников по фолдам (с кэшем) и ансамбль LightGBM + CatBoost из готовых прогнозов."""
    raws: dict[str, dict[str, RawFold]] = {}
    for fold in fold_list:
        train, test = BT.split(rows, fold)
        for name, learner in learners(features).items():
            path = RAW / f"{name}_{fold.name}_{_key(name, learner, fold, len(train))}.pkl"
            if (reuse or name == "lgbm") and path.exists():      # LightGBM — всегда из кэша, без переобучения
                raw = pickle.loads(path.read_bytes())
            else:
                raw = fit_raw(learner, train, test)
                RAW.mkdir(parents=True, exist_ok=True)
                path.write_bytes(pickle.dumps(raw))
            raws.setdefault(name, {})[fold.name] = raw
            log(f"  {fold.name}: {name} — {raw.seconds:.0f} с, итераций {raw.iters}", flush=True)
        for name, raw in ensembles({n: r[fold.name] for n, r in raws.items()}, train).items():
            raws.setdefault(name, {})[fold.name] = raw
            log(f"  {fold.name}: {name} — веса {raw.iters['weights']}", flush=True)
    return raws


def predictions(rows: pd.DataFrame, fold_list: list[BT.Fold],
                raws: dict[str, dict[str, RawFold]]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Калиброванные прогнозы в формате BT.run и сведения о калибровке по фолдам."""
    preds, infos = {}, []
    for name, by_fold in raws.items():
        parts = []
        for fold in fold_list:
            train, test = BT.split(rows, fold)
            raw = by_fold[fold.name]
            p, info = calibrate(raw, train, test)
            parts.append(test.loc[raw.test_index, PRED_COLS].assign(model=name, fold=fold.name).join(p))
            infos.append({"model": name, "fold": fold.name, "seconds": raw.seconds, "iters": json.dumps(
                {str(k): v for k, v in raw.iters.items()}, default=str), **info})
        preds[name] = pd.concat(parts, ignore_index=True)
    return preds, pd.DataFrame(infos)


def check_reference(lgbm: pd.DataFrame, frozen: pd.DataFrame, tol: float = 1e-6) -> float:
    """LightGBM внутри общей схемы обязан совпасть с замороженными прогнозами: те же строки, веса, окна, калибровка."""
    M._check_aligned(frozen, lgbm)
    diff = max(float(np.abs(lgbm[c].to_numpy() - frozen[c].to_numpy()).max()) for c in ("q10", "q50", "q90"))
    if diff > tol:
        raise RuntimeError(f"эталон LightGBM не совпал с замороженной моделью: max |Δ| = {diff:.3g}")
    return diff


def summarize(preds: dict[str, pd.DataFrame], calib: pd.DataFrame, ref: str = "lgbm") -> pd.DataFrame:
    """По модели: WAPE по срезам и фолдам, покрытие, ΔWAPE к LightGBM [ДИ], p DM (все и аномальные часы), время."""
    base = preds[ref]
    out = []
    for name, p in preds.items():
        s = M.summary(p)
        res = BT.evaluate(p)
        allh = res[(res.fold == "все") & (res.slice == "все часы")]
        row = {"model": name, "label": LABELS[name], **s, "pinball": allh.pinball.mean(),
               "pinball_norm": allh.pinball_norm.mean(), "seconds_per_fold": calib[calib.model == name].seconds.mean()}
        if name != ref:
            row.update({f"d_{k}": v for k, v in M.compare(base, p).items()})
            row.update({f"danom_{k}": v for k, v in M.compare(base, p, base.is_anomaly).items()})
            fd = M.fold_delta(base, p)                    # > 0 — модель лучше LightGBM в этом фолде
            row.update({f"dfold_{f}": v for f, v in fd.items()})
            row["folds_better"] = int((fd > 0).sum())
            row["n_folds"] = int(len(fd))
            better = row["d_dwape"] > 0 and row["d_lo"] > 0 and row["folds_better"] >= min(MIN_FOLDS, len(fd))
            row["verdict"] = ("кандидат во вторую версию" if better else
                              "не отличима от LightGBM" if row["d_hi"] > 0 else "хуже LightGBM")
        else:
            row["verdict"] = "официальная (заморожена)"
        out.append(row)
    return pd.DataFrame(out)


def _pct(x, nd=2):
    return "—" if pd.isna(x) else f"{x * 100:.{nd}f} %".replace(".", ",")


def _pp(x):
    return "—" if pd.isna(x) else ("−" if x < 0 else "+") + f"{abs(x) * 100:.2f}".replace(".", ",")


def _p(x):
    return "—" if pd.isna(x) else ("< 0,001" if x < 0.001 else f"{x:.3f}".replace(".", ","))


def markdown(summ: pd.DataFrame) -> str:
    lines = ["| Модель | WAPE | t+1 / t+2 | Пики | Аномальные | Покрытие | Pinball | ΔWAPE к LightGBM, п. п. [95 % ДИ] | "
             "p DM | Фолдов лучше | Обучение, с на фолд | Вывод |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in summ.itertuples():
        d = "—" if r.model == "lgbm" else f"{_pp(r.d_dwape)} [{_pp(r.d_lo)}; {_pp(r.d_hi)}]"
        p = "—" if r.model == "lgbm" else _p(r.d_p_dm)
        folds = "—" if r.model == "lgbm" else f"{r.folds_better} из {r.n_folds}"
        pin = f"{r.pinball:.2f}".replace(".", ",")
        lines.append(f"| {r.label} | {_pct(r.wape)} | {_pct(r.wape_h1)} / {_pct(r.wape_h2)} | {_pct(r.wape_peaks)} | "
                     f"{_pct(r.wape_anomaly)} | {_pct(r.coverage, 1)} | {pin} | {d} | {p} | {folds} | "
                     f"{r.seconds_per_fold:.0f} | {r.verdict} |")
    return "\n".join(lines)


def run(rows, fold_list, cfg, reuse=False, check_frozen_pred: pd.DataFrame | None = None, log=print):
    raws = run_raw(rows, fold_list, cfg["features"], reuse=reuse, log=log)
    preds, calib = predictions(rows, fold_list, raws)
    diff = check_reference(preds["lgbm"], check_frozen_pred) if check_frozen_pred is not None else None
    summ = summarize(preds, calib)
    return preds, calib, summ, diff


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reuse", action="store_true", help="сырые прогнозы — из кэша, если признаки и параметры те же")
    ap.add_argument("--september", action="store_true",
                    help="один прогон всех моделей на сентябре «для информации» (после выбора по маю–августу)")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = M.load_config()
    M.check_frozen(cfg)                               # замороженная конфигурация и код не менялись

    if args.september:
        if not (OUT / "summary.csv").exists():
            raise SystemExit("сначала выбор по маю–августу: python -m src.compare_models")
        done = OUT / "september.csv"
        if done.exists() and set(pd.read_csv(done).model) >= set(LABELS):
            raise SystemExit("прогон на сентябре всех моделей уже сделан: reports/compare/september.csv")
        panel = M.frozen_panel(cfg, final=True)
        rows = F.add_features(BT.make_rows(panel), panel)
        fold_list = [BT.get_fold("final", final=True)]
        preds, calib, summ, _ = run(rows, fold_list, cfg, reuse=args.reuse)
        frozen = pd.read_csv(config.ROOT / "reports" / "backtest" / "final_results.csv")
        w_frozen = frozen[(frozen.model == "lgbm_final") & (frozen.fold == "все") & (frozen.slice == "все часы")].wape.mean()
        w_ref = summ.set_index("model").loc["lgbm", "wape"]
        if abs(w_frozen - w_ref) > 1e-6:          # final_results.csv сохранён с 6 знаками
            raise RuntimeError(f"эталон LightGBM на сентябре: {w_ref:.6f} против финального теста {w_frozen:.6f}")
        # выбор делается только по маю–августу: на сентябре вердиктов нет, таблица — «для информации»
        summ["verdict"] = np.where(summ.model == "lgbm", "официальная (заморожена)", "для информации")
        BT.evaluate(pd.concat(preds.values(), ignore_index=True)).round(6).to_csv(OUT / "september_results.csv",
                                                                                    index=False)
        summ.round(6).to_csv(OUT / "september.csv", index=False)
        calib.round(5).to_csv(OUT / "september_calibration.csv", index=False)
        (OUT / "september.md").write_text(markdown(summ) + "\n", encoding="utf-8")
        print(f"эталон LightGBM совпал с финальным тестом: WAPE {w_ref:.6f}")
        print(markdown(summ))
        return

    panel = M.frozen_panel(cfg, final=False)
    rows = F.add_features(BT.make_rows(panel), panel)
    fold_list = BT.folds()
    frozen = M.run_config(rows, M.make_model(cfg["features"], cfg["target"], cfg["step_model"]), fold_list,
                          reuse=True)[0]
    preds, calib, summ, diff = run(rows, fold_list, cfg, reuse=args.reuse, check_frozen_pred=frozen)
    BT.evaluate(pd.concat(preds.values(), ignore_index=True)).round(6).to_csv(OUT / "results.csv", index=False)
    summ.round(6).to_csv(OUT / "summary.csv", index=False)
    calib.round(5).to_csv(OUT / "calibration.csv", index=False)
    (OUT / "summary.md").write_text(markdown(summ) + "\n", encoding="utf-8")
    print(f"эталон LightGBM совпал с замороженными прогнозами: max |Δ| = {diff:.2g}")
    print(markdown(summ))


if __name__ == "__main__":
    main()
