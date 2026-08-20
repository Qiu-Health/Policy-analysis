from pathlib import Path

import pandas as pd
import numpy as np
import statsmodels.api as sm  # type: ignore[reportMissingImports]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, norm  # type: ignore[reportMissingImports]
from patsy import build_design_matrices  # type: ignore[reportMissingImports]
from statsmodels.stats.diagnostic import het_breuschpagan  # type: ignore[reportMissingImports]
from statsmodels.stats.stattools import durbin_watson  # type: ignore[reportMissingImports]
from statsmodels.tsa.stattools import adfuller  # type: ignore[reportMissingImports]
from scipy.stats import zscore  # type: ignore[reportMissingImports]
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, acorr_ljungbox  # type: ignore[reportMissingImports]

RESULTS_OUTPUT_DIR = Path(r'D:\检测小队\林嘉意\中断时间序列\Global\Part. 2\results output')
# 1. Data preprocessing (based on the provided code)
df1 = pd.read_csv(r'D:\检测小队\林嘉意\中断时间序列\Global\data\PFOS_NAH.csv',
                    usecols=['year', 'month', 'Substance', 'lon', 'lat', 'lon_grid', 'lat_grid',
                             'country', 'value [ng/L]', 'source', 'limit [ng/L]'],
                    dtype={'value [ng/L]': float})

# # Retain only data before 2016 (excluding 2016)
# df1 = df1[df1['year'] < 2015]

# Fill missing values in the 'source' column
df1['source'] = df1['source'].fillna('Unknown')
# 1) Fill missing limit [ng/L] values with 0.003536; replace zero value [ng/L] values with missing values
df1['limit [ng/L]'] = df1['limit [ng/L]'].fillna(0.003536)
df1.loc[df1['value [ng/L]'] == 0, 'value [ng/L]'] = np.nan

# 2) Remove rows where limit [ng/L] >= 1 and value [ng/L] is missing
df1 = df1[~((df1['limit [ng/L]'] >= 1) & (df1['value [ng/L]'].isna()))].copy()

# 3) When value [ng/L] is missing, fill it with limit [ng/L] / sqrt(2) from the same row
missing_value_mask = df1['value [ng/L]'].isna()
df1.loc[missing_value_mask, 'value [ng/L]'] = df1.loc[missing_value_mask, 'limit [ng/L]'] / np.sqrt(2)
#  df1.loc[missing_value_mask, 'value [ng/L]'] = 0.003536 / np.sqrt(2)

# 2. Calculate grouped medians
median_raw = df1.groupby(['year', 'month', 'source', 'lon', 'lat', 'lon_grid', 'lat_grid'])[
                'value [ng/L]'].median().reset_index()
median_cell = median_raw.groupby(['year', 'month', 'source', 'lon_grid', 'lat_grid'])[
                'value [ng/L]'].median().reset_index()
median_ref = median_cell.groupby(['year', 'month', 'lon_grid', 'lat_grid'])[
    'value [ng/L]'].median().reset_index()
median_year_month = median_ref.groupby(['year', 'month'])[
    'value [ng/L]'].median().reset_index()
# Save the results to files
median_ref.to_excel(RESULTS_OUTPUT_DIR / "global_monthly_cell_PFOS.xlsx", index=False)
median_year_month.to_excel(RESULTS_OUTPUT_DIR / "global_monthly_median_PFOS.xlsx", index=False)

# 3. Detect outliers in the original data, then construct a complete time series by linear interpolation
# =========================

# 3.1 Prepare the original time-series data (without interpolation)
mm = median_year_month.copy()

# Construct year_month
# Ensure integer values and remove decimal components
mm['year'] = mm['year'].astype(int)
mm['month'] = mm['month'].astype(int)

# Build a standard string and then convert it to a date
mm['year_month'] = pd.to_datetime(
    mm['year'].astype(str) + '-' + mm['month'].astype(str).str.zfill(2),
    format='%Y-%m'
)
mm = mm.sort_values('year_month').drop_duplicates(subset=['year_month'])

# Generate a complete monthly index for subsequent interpolation
full_index = pd.date_range(start=mm['year_month'].min(),
                           end=mm['year_month'].max(),
                           freq='MS')

# 3.2 Perform linear fitting and outlier detection on the original observations (without interpolation)
print("基于原始观测数据进行异常值检测...")

# Fit a linear model using the original observations
mm_original = mm.copy()
mm_original['time'] = mm_original['year'] + (mm_original['month'] / 12.0)  # Continuous time

# Fit only at points with observed values
observed_mask = ~mm_original['value [ng/L]'].isna()
X_observed = sm.add_constant(mm_original.loc[observed_mask, 'time'])
y_observed = mm_original.loc[observed_mask, 'value [ng/L]']

model_original = sm.OLS(y_observed, X_observed).fit()

# Calculate residuals for all time points (including positions with missing values)
mm_original['residuals'] = np.nan
mm_original.loc[observed_mask, 'residuals'] = model_original.resid

# Standardize residuals as Z-scores (based only on observed points)
residuals_observed = mm_original.loc[observed_mask, 'residuals']
std_residuals_observed = zscore(residuals_observed)
mm_original.loc[observed_mask, 'std_residuals'] = std_residuals_observed

# Flag outliers (|Z| > 3, only at points with observed values)
mm_original['is_outlier'] = False
mm_original.loc[observed_mask, 'is_outlier'] = mm_original.loc[observed_mask, 'std_residuals'].abs() > 3

# Output the outlier-detection results
outliers_original = mm_original.loc[mm_original['is_outlier'], ['year', 'month', 'year_month', 'value [ng/L]', 'residuals', 'std_residuals']]
print("基于原始观测数据检测到的异常值：")
print(outliers_original)

# Save the outlier list
out_path_outliers = RESULTS_OUTPUT_DIR / "global_PFOS_outliers_before_interp.xlsx"
outliers_original.to_excel(out_path_outliers, index=False)

# 3.3 Remove outliers and perform one-pass linear interpolation
print("\n删除异常值并进行一次性线性插值...")

# Remove outliers and retain normal observations
mm_clean_original = mm_original[~mm_original['is_outlier']].copy()

# Mark the original observations after outlier removal
mm_clean_original['observed'] = 1

# Create a complete time-series frame
complete_ts = pd.DataFrame({'year_month': full_index})
complete_ts['year'] = complete_ts['year_month'].dt.year
complete_ts['month'] = complete_ts['year_month'].dt.month

# Merge the cleaned observations
complete_ts = pd.merge(complete_ts,
                      mm_clean_original[['year_month', 'value [ng/L]', 'observed']],
                      on='year_month', how='left')

# Mark which points are original observations after outlier removal
complete_ts['observed'] = complete_ts['observed'].fillna(0).astype(int)

# Perform one-pass linear interpolation
print(f"插值前缺失值数量: {complete_ts['value [ng/L]'].isna().sum()}")
complete_ts['value [ng/L]'] = complete_ts['value [ng/L]'].interpolate(method='linear')

# Check for remaining missing values at the start or end of the series; use backward/forward fill if needed
if complete_ts['value [ng/L]'].isna().any():
    print("序列开头或结尾仍有缺失值，使用前向/后向填充...")
    complete_ts['value [ng/L]'] = complete_ts['value [ng/L]'].fillna(method='bfill').fillna(method='ffill')

# 3.4 Save the results
data = complete_ts.copy()

# ====== Additional statistical tests: Mann-Whitney U / ADF / Breusch-Pagan / Durbin-Watson ======
median_year_month['date'] = pd.to_datetime(
    median_year_month['year'].astype(str) + '-' + median_year_month['month'].astype(str).str.zfill(2) + '-01'
)

cutoff_date = pd.to_datetime('2010-08-01')
group_before = median_year_month.loc[median_year_month['date'] < cutoff_date, 'value [ng/L]']
group_after = median_year_month.loc[median_year_month['date'] >= cutoff_date, 'value [ng/L]']

median_before = group_before.median()
median_after = group_after.median()

# Traditional Mann-Whitney (non-parametric) — note: assumes independence
mw_stat, mw_p = mannwhitneyu(group_before.dropna(), group_after.dropna(), alternative='two-sided')

print("=== Mann–Whitney U 检验（传统） ===")
print(f"2010年8月前 中位数: {median_before:.3f} ng/L, 样本量: {len(group_before.dropna())}")
print(f"2010年8月后 中位数: {median_after:.3f} ng/L, 样本量: {len(group_after.dropna())}")
print(f"U 统计量: {mw_stat:.3f}, p 值: {mw_p:.5f}")

print('\n注：时间序列观测常存在自相关，Mann-Whitney 的独立性假设可能被违反。下面同时输出基于 block-bootstrap 的中位数差检验（更保留时序信息）。')

# Block bootstrap for median difference (preserve temporal dependence)
def block_bootstrap_median_diff(series: pd.Series, dates: pd.Series, cutoff: pd.Timestamp, n_boot=1000, block_size=3, seed=42):
    rng = np.random.default_rng(seed)
    series = series.reset_index(drop=True)
    dates = dates.reset_index(drop=True)
    n = len(series)
    starts = np.arange(0, n - block_size + 1)
    pre_idx = dates < cutoff
    post_idx = ~pre_idx
    obs_pre_med = np.nanmedian(series[pre_idx.values])
    obs_post_med = np.nanmedian(series[post_idx.values])
    obs_diff = obs_post_med - obs_pre_med
    boot_diffs = []
    for _ in range(n_boot):
        res = []
        while len(res) < n:
            s = int(rng.choice(starts))
            res.extend(series[s:s+block_size].tolist())
        res = np.array(res[:n])
        boot_pre_med = np.nanmedian(res[pre_idx.values])
        boot_post_med = np.nanmedian(res[post_idx.values])
        boot_diffs.append(boot_post_med - boot_pre_med)
    boot_diffs = np.array(boot_diffs)
    p_boot = np.mean(np.abs(boot_diffs) >= np.abs(obs_diff))
    return obs_diff, p_boot, boot_diffs

obs_series = data['value [ng/L]'].copy()
obs_dates = data['year_month'].copy()
obs_diff, p_boot, boot_dist = block_bootstrap_median_diff(obs_series, obs_dates, cutoff_date, n_boot=1000, block_size=3)
print('\n=== Block-bootstrap (median diff) ===')
print(f'Intervention 后-前 中位数差: {obs_diff:.5f} ng/L, bootstrap p-value: {p_boot:.4f}')


def safe_adf(series: pd.Series, name: str):
    s = series.dropna()
    print(f"\n【ADF 检验 - {name}】 样本量: {len(s)}")
    if len(s) < 10:
        print("样本过少，ADF 结果不可靠（建议至少 ~10 个观测点）。")
        return None
    try:
        res = adfuller(s)
        print(f"ADF 统计量: {res[0]:.4f}, p 值: {res[1]:.4f}")
        print("序列为" + ("非平稳" if res[1] > 0.05 else "平稳"))
        return res
    except Exception as e:
        print('ADF 检验出错:', e)
        return None

# ADF on interpolated full series (data) and on observed-only aggregated series (median_year_month)
safe_adf(data.loc[data['year_month'] < cutoff_date, 'value [ng/L]'], 'Interpolated - Pre')
safe_adf(data.loc[data['year_month'] >= cutoff_date, 'value [ng/L]'], 'Interpolated - Post')
safe_adf(median_year_month.loc[median_year_month['date'] < cutoff_date, 'value [ng/L]'], 'Observed aggregated - Pre')
safe_adf(median_year_month.loc[median_year_month['date'] >= cutoff_date, 'value [ng/L]'], 'Observed aggregated - Post')

def check_heteroscedasticity(model, name):
    lm, lm_pvalue, fvalue, f_pvalue = het_breuschpagan(model.resid, model.model.exog)
    print(f"\n【Breusch-Pagan 异方差检验 - {name}】")
    print(f"LM统计量: {lm:.4f}, p值: {lm_pvalue:.4f}")
    print(f"F统计量: {fvalue:.4f}, p值: {f_pvalue:.4f}")
    has_heteroscedasticity = lm_pvalue < 0.05
    if lm_pvalue < 0.05:
        print("存在异方差性")
        try:
            robust = model.get_robustcov_results(cov_type='HC3')
            print(f"使用 HC3 估计的标准误（简要）：\n{robust.summary().tables[1]}")
        except Exception:
            print('生成 HC3 稳健标准误失败。')
    else:
        print("未发现显著异方差性")
    return {
        'lm_pvalue': lm_pvalue,
        'f_pvalue': f_pvalue,
        'has_heteroscedasticity': has_heteroscedasticity,
    }

def check_dw(model, name):
    dw = durbin_watson(model.resid)
    print(f"\n【Durbin-Watson 检验 - {name}】")
    print(f"DW 值: {dw:.4f}")
    has_autocorrelation = dw < 1.5 or dw > 2.5
    if dw < 1.5:
        print("存在正自相关")
    elif dw > 2.5:
        print("存在负自相关")
    else:
        print("未发现严重自相关")
    # Breusch-Godfrey (higher-order autocorrelation)
    try:
        bg = acorr_breusch_godfrey(model, nlags=4)
        print(f"\n【Breusch-Godfrey (Lags=4) - {name}】 LM stat: {bg[0]:.4f}, p: {bg[1]:.4f}")
        bg_pvalue = bg[1]
        has_autocorrelation = has_autocorrelation or (bg_pvalue < 0.05)
    except Exception:
        print('Breusch-Godfrey 检验失败。')
        bg_pvalue = np.nan
    # Ljung-Box on residuals
    try:
        lb = acorr_ljungbox(model.resid, lags=[12], return_df=True)
        print(f"\n【Ljung-Box (lag=12) - {name}】\n{lb}")
        lb_pvalue = float(lb['lb_pvalue'].iloc[-1])
        has_autocorrelation = has_autocorrelation or (lb_pvalue < 0.05)
    except Exception:
        print('Ljung-Box 检验失败。')
        lb_pvalue = np.nan
    # Recommend HAC if autocorrelation found
    if dw < 1.5 or (('bg' in locals()) and bg[1] < 0.05):
        print('检测到自相关，建议使用 Newey-West (HAC) 稳健 SE 或使用 SARIMAX/AR 模型（已提供 SARIMAX 敏感性）。')
    return {
        'dw': dw,
        'bg_pvalue': bg_pvalue,
        'lb_pvalue': lb_pvalue,
        'has_autocorrelation': has_autocorrelation,
    }


def choose_final_cov_type(bp_info: dict, dw_info: dict) -> str:
    """Choose OLS / HC3 / HAC based on heteroscedasticity and autocorrelation checks."""
    if dw_info.get('has_autocorrelation', False):
        return 'HAC'
    if bp_info.get('has_heteroscedasticity', False):
        return 'HC3'
    return 'OLS'


def get_final_model_results(model, cov_type: str, maxlags: int = 4):
    if cov_type == 'HC3':
        return model.get_robustcov_results(cov_type='HC3')
    if cov_type == 'HAC':
        return model.get_robustcov_results(cov_type='HAC', maxlags=maxlags)
    return model

def save_diagnostic_tables(models: dict[str, object], out_prefix: Path):
    rows = []
    for name, model in models.items():
        params = model.params
        pvals = model.pvalues
        try:
            nw = model.get_robustcov_results(cov_type='HAC', maxlags=4)
        except Exception:
            nw = None
        nw_se = None
        if nw is not None:
            nw_se = pd.Series(np.asarray(nw.bse), index=list(model.params.index))
        rows.append({
            'Model': name,
            'AIC': model.aic,
            'BIC': model.bic,
            'LogLik': model.llf,
            'Intercept': params.get('Intercept', np.nan),
            'Pre_trend_time': params.get('time', np.nan),
            'Level_change_period': params.get('period', np.nan),
            'Slope_change_time_after': params.get('time_after', np.nan),
            'P_value_period': pvals.get('period', np.nan),
            'P_value_time_after': pvals.get('time_after', np.nan),
            'NW_se_period': nw_se.get('period', np.nan) if nw_se is not None else np.nan,
            'NW_se_time_after': nw_se.get('time_after', np.nan) if nw_se is not None else np.nan,
        })
    df = pd.DataFrame(rows)
    print('\n=== 诊断表摘要（仅 terminal 输出）===')
    print(df.round(4).to_string(index=False))

intervention_date = pd.to_datetime('2010-08-01')


def build_itsa_frame(frame: pd.DataFrame) -> pd.DataFrame:
    itsa_frame = frame.copy().sort_values('year_month').reset_index(drop=True)
    itsa_frame['time'] = np.arange(len(itsa_frame))
    itsa_frame['period'] = (itsa_frame['year_month'] >= intervention_date).astype(int)
    intervention_time = itsa_frame.loc[itsa_frame['year_month'] == intervention_date, 'time']
    if intervention_time.empty:
        raise ValueError('干预日期不在时间序列中，请检查 intervention_date。')
    t0 = int(intervention_time.iloc[0])
    itsa_frame['time_after'] = np.where(itsa_frame['period'] == 1, itsa_frame['time'] - t0, 0)
    return itsa_frame


def fit_ols_models(frame: pd.DataFrame) -> dict[str, object]:
    print('\n开始拟合 OLS ITSA 主分析（线性插值）...')
    models = {
        'Level Change Only': sm.OLS.from_formula('Q("value [ng/L]") ~ time + period', data=frame).fit(),
        'Slope Change Only': sm.OLS.from_formula('Q("value [ng/L]") ~ time + time_after', data=frame).fit(),
        'Level + Slope Change': sm.OLS.from_formula('Q("value [ng/L]") ~ time + period + time_after', data=frame).fit(),
    }
    for name, model in models.items():
        print(f'\n=== {name} ===')
        print(model.summary())
    return models


def ci_bounds(conf_int_result: object) -> tuple[pd.Series, pd.Series]:
    conf_array = np.asarray(conf_int_result)
    return pd.Series(conf_array[:, 0]), pd.Series(conf_array[:, 1])


def _as_named_series(values: object, index: pd.Index) -> pd.Series:
    return pd.Series(np.asarray(values), index=index)


def _robust_results(model: object, cov_type: str, maxlags: int | None = None):
    if maxlags is None:
        return model.get_robustcov_results(cov_type=cov_type)
    return model.get_robustcov_results(cov_type=cov_type, maxlags=maxlags)


def _prediction_exog(model: object, frame: pd.DataFrame) -> np.ndarray:
    """Build the exact design matrix for a new frame when the model was fit via formula."""
    design_info = getattr(getattr(model.model, 'data', None), 'design_info', None)
    if design_info is not None:
        return np.asarray(build_design_matrices([design_info], frame)[0])
    return np.asarray(getattr(model.model, 'exog'))


def fit_sarimax_with_retry(
    endog: pd.Series,
    exog: pd.DataFrame,
    order: tuple[int, int, int],
    label: str,
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
):
    """Fit SARIMAX with a small retry ladder to reduce convergence warnings."""
    fit_attempts = [
        {'method': 'lbfgs', 'maxiter': 2000},
        {'method': 'powell', 'maxiter': 3000},
        {'method': 'nm', 'maxiter': 3000},
    ]
    last_error = None
    for attempt in fit_attempts:
        try:
            mod = sm.tsa.SARIMAX(
                endog,
                exog=exog,
                order=order,
                seasonal_order=seasonal_order,
                trend='c',
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            res = mod.fit(disp=False, **attempt)
            converged = bool(getattr(res, 'mle_retvals', {}).get('converged', True))
            if not converged:
                print(f'SARIMAX {label} ({order}, seasonal={seasonal_order}) 未完全收敛，当前方法={attempt["method"]}；将尝试下一种优化器。')
                last_error = RuntimeError('SARIMAX did not converge')
                continue
            return res
        except Exception as e:
            last_error = e
            print(f'SARIMAX {label} ({order}, seasonal={seasonal_order}) 使用 {attempt["method"]} 拟合失败：{e}')
    raise RuntimeError(f'SARIMAX {label} ({order}, seasonal={seasonal_order}) 多次拟合仍未收敛：{last_error}')


def compute_mean_ci(model: object, frame: pd.DataFrame, choice: str = 'HAC', maxlags: int = 4, alpha: float = 0.05):
    """Compute mean prediction CI using OLS / HC3 / HAC according to choice."""
    # predicted mean (uses model.predict which applies the model params)
    pred_mean = model.predict(frame)

    choice_up = (choice or 'OLS').upper()
    if choice_up == 'HAC':
        try:
            robust = _robust_results(model, cov_type='HAC', maxlags=maxlags)
            cov = robust.cov_params()
            cov_arr = cov.values if hasattr(cov, 'values') else np.asarray(cov)
        except Exception:
            cov_arr = model.cov_params().values if hasattr(model.cov_params(), 'values') else np.asarray(model.cov_params())
    elif choice_up == 'HC3':
        try:
            robust = _robust_results(model, cov_type='HC3')
            cov = robust.cov_params()
            cov_arr = cov.values if hasattr(cov, 'values') else np.asarray(cov)
        except Exception:
            cov_arr = model.cov_params().values if hasattr(model.cov_params(), 'values') else np.asarray(model.cov_params())
    else:
        cov_arr = model.cov_params().values if hasattr(model.cov_params(), 'values') else np.asarray(model.cov_params())

    # build the correct exog for the provided frame (important for counterfactuals)
    exog = _prediction_exog(model, frame)

    # variance for the mean prediction for each row: x_i' cov x_i
    se_mean = np.sqrt(np.einsum('ij,jk,ik->i', exog, cov_arr, exog))
    z = -1 * norm.ppf(alpha / 2)
    lower = pred_mean - z * se_mean
    upper = pred_mean + z * se_mean
    return pred_mean, lower, upper


def compute_mean_hac_ci(model: object, frame: pd.DataFrame, maxlags: int = 4, alpha: float = 0.05):
    return compute_mean_ci(model, frame, choice='HAC', maxlags=maxlags, alpha=alpha)


def _safe_get(series: pd.Series, key: str) -> float:
    return float(series.get(key, np.nan)) if key in series.index else np.nan


def ols_results_table(models: dict[str, object]) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        params = model.params
        pvalues = _as_named_series(model.pvalues, params.index)
        conf = model.conf_int()
        try:
            hc3 = _robust_results(model, cov_type='HC3')
            hc3_bse = _as_named_series(hc3.bse, params.index)
            hc3_pvalues = _as_named_series(hc3.pvalues, params.index)
            hc3_conf = pd.DataFrame(np.asarray(hc3.conf_int()), index=params.index)
        except Exception:
            hc3_bse = params * np.nan
            hc3_pvalues = params * np.nan
            hc3_conf = pd.DataFrame(np.nan, index=params.index, columns=[0, 1])

        try:
            hac = _robust_results(model, cov_type='HAC', maxlags=4)
            hac_bse = _as_named_series(hac.bse, params.index)
            hac_pvalues = _as_named_series(hac.pvalues, params.index)
            hac_conf = pd.DataFrame(np.asarray(hac.conf_int()), index=params.index)
        except Exception:
            hac_bse = params * np.nan
            hac_pvalues = params * np.nan
            hac_conf = pd.DataFrame(np.nan, index=params.index, columns=[0, 1])

        rows.append({
            'Model': name,
            'AIC': model.aic,
            'BIC': model.bic,
            'LogLik': model.llf,
            'Intercept': params.get('Intercept', np.nan),
            'Pre_trend_time': params.get('time', np.nan),
            'Level_change_period': params.get('period', np.nan),
            'Slope_change_time_after': params.get('time_after', np.nan),
            'SE_ordinary_Intercept': _safe_get(model.bse, 'Intercept'),
            'SE_ordinary_time': _safe_get(model.bse, 'time'),
            'SE_ordinary_period': _safe_get(model.bse, 'period'),
            'SE_ordinary_time_after': _safe_get(model.bse, 'time_after'),
            'P_ordinary_Intercept': _safe_get(pvalues, 'Intercept'),
            'P_ordinary_time': _safe_get(pvalues, 'time'),
            'P_ordinary_period': _safe_get(pvalues, 'period'),
            'P_ordinary_time_after': _safe_get(pvalues, 'time_after'),
            'SE_HC3_Intercept': _safe_get(hc3_bse, 'Intercept'),
            'SE_HC3_time': _safe_get(hc3_bse, 'time'),
            'SE_HC3_period': _safe_get(hc3_bse, 'period'),
            'SE_HC3_time_after': _safe_get(hc3_bse, 'time_after'),
            'P_HC3_Intercept': _safe_get(hc3_pvalues, 'Intercept'),
            'P_HC3_time': _safe_get(hc3_pvalues, 'time'),
            'P_HC3_period': _safe_get(hc3_pvalues, 'period'),
            'P_HC3_time_after': _safe_get(hc3_pvalues, 'time_after'),
            'SE_HAC_Intercept': _safe_get(hac_bse, 'Intercept'),
            'SE_HAC_time': _safe_get(hac_bse, 'time'),
            'SE_HAC_period': _safe_get(hac_bse, 'period'),
            'SE_HAC_time_after': _safe_get(hac_bse, 'time_after'),
            'P_HAC_Intercept': _safe_get(hac_pvalues, 'Intercept'),
            'P_HAC_time': _safe_get(hac_pvalues, 'time'),
            'P_HAC_period': _safe_get(hac_pvalues, 'period'),
            'P_HAC_time_after': _safe_get(hac_pvalues, 'time_after'),
            'P_value_period': model.pvalues.get('period', np.nan),
            'P_value_time_after': model.pvalues.get('time_after', np.nan),
            'CI_low_period': conf.loc['period', 0] if 'period' in conf.index else np.nan,
            'CI_high_period': conf.loc['period', 1] if 'period' in conf.index else np.nan,
            'CI_low_time_after': conf.loc['time_after', 0] if 'time_after' in conf.index else np.nan,
            'CI_high_time_after': conf.loc['time_after', 1] if 'time_after' in conf.index else np.nan,
            'CI_HC3_low_period': hc3_conf.loc['period', 0] if 'period' in hc3_conf.index else np.nan,
            'CI_HC3_high_period': hc3_conf.loc['period', 1] if 'period' in hc3_conf.index else np.nan,
            'CI_HC3_low_time_after': hc3_conf.loc['time_after', 0] if 'time_after' in hc3_conf.index else np.nan,
            'CI_HC3_high_time_after': hc3_conf.loc['time_after', 1] if 'time_after' in hc3_conf.index else np.nan,
            'CI_HAC_low_period': hac_conf.loc['period', 0] if 'period' in hac_conf.index else np.nan,
            'CI_HAC_high_period': hac_conf.loc['period', 1] if 'period' in hac_conf.index else np.nan,
            'CI_HAC_low_time_after': hac_conf.loc['time_after', 0] if 'time_after' in hac_conf.index else np.nan,
            'CI_HAC_high_time_after': hac_conf.loc['time_after', 1] if 'time_after' in hac_conf.index else np.nan,
        })
    return pd.DataFrame(rows)


def save_itsa_plot(frame: pd.DataFrame, model: object, title: str, output_name: str, ci_choice: str = 'HAC') -> None:
    fig, ax = plt.subplots(figsize=(16, 8))
    observed_data = frame[frame['observed'] == 1]
    # Use OLS point predictions but diagnosis-selected CI for the mean prediction
    fitted, lower, upper = compute_mean_ci(model, frame, choice=ci_choice, maxlags=4)
    # counterfactual (no intervention)
    counterfactual = frame.copy()
    counterfactual['period'] = 0
    counterfactual['time_after'] = 0
    fitted_cf, lower_cf, upper_cf = compute_mean_ci(model, counterfactual, choice=ci_choice, maxlags=4)

    ax.scatter(frame['year_month'], frame['value [ng/L]'], s=56, color='grey', label='Interpolated series', alpha=0.45)
    ax.scatter(observed_data['year_month'], observed_data['value [ng/L]'], s=80, color='black', label='Observed median', alpha=1, edgecolors='black', linewidths=0.5, zorder=3)
    ax.plot(frame['year_month'], fitted, 'b-', lw=2, label='Fitted trend (OLS)')
    ci_label = ci_choice.upper()
    ax.fill_between(frame['year_month'], lower, upper, color='blue', alpha=0.18, label=f'95% CI ({ci_label} mean CI)')
    ax.plot(frame['year_month'], fitted_cf, 'r--', lw=2, label='Counterfactual (no intervention)')
    ax.fill_between(frame['year_month'], lower_cf, upper_cf, color='red', alpha=0.18, label=f'95% CI ({ci_label} mean CI CF)')
    ax.axvline(x=intervention_date, color='k', linestyle='--', alpha=0.6, label='Intervention (Aug 2010)')
    ax.set_title(title + '\nGlobal PFOS Concentration Analysis', fontsize=14)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('PFOS Concentration [ng/L]', fontsize=12)
    xticks = frame[frame['month'] == 1]['year_month']
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks.dt.year, rotation=45)
    ax.set_xlim(frame['year_month'].min(), frame['year_month'].max())
    ax.set_ylim(frame['value [ng/L]'].min() * 0.9, frame['value [ng/L]'].max() * 1.1)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    fig.tight_layout()
    out_path = RESULTS_OUTPUT_DIR / output_name
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'ITSA 图已保存到：{out_path}')


def fit_sarimax_sensitivity(frame: pd.DataFrame) -> tuple[object, pd.DataFrame]:
    print('\n开始拟合 SARIMAX 敏感性分析（保留缺失值，不插值）...')
    sensitivity_frame = pd.DataFrame({'year_month': full_index})
    sensitivity_frame['year'] = sensitivity_frame['year_month'].dt.year
    sensitivity_frame['month'] = sensitivity_frame['year_month'].dt.month
    sensitivity_frame = pd.merge(
        sensitivity_frame,
        mm_clean_original[['year_month', 'value [ng/L]', 'observed']],
        on='year_month',
        how='left',
    )
    sensitivity_frame['observed'] = sensitivity_frame['observed'].fillna(0).astype(int)
    sensitivity_frame = build_itsa_frame(sensitivity_frame)

    endog = sensitivity_frame['value [ng/L]']
    exog = sensitivity_frame[['time', 'period', 'time_after']]
    sarimax_res = fit_sarimax_with_retry(endog, exog, (1, 0, 1), 'Sensitivity')
    print('\n=== SARIMAX Sensitivity Summary ===')
    print(sarimax_res.summary())

    pred = sarimax_res.get_prediction(start=0, end=len(sensitivity_frame) - 1, exog=exog)
    sensitivity_frame['sarimax_fitted'] = pred.predicted_mean
    sarimax_ci = pred.conf_int()
    sensitivity_frame['sarimax_ci_lower'], sensitivity_frame['sarimax_ci_upper'] = ci_bounds(sarimax_ci)
    return sarimax_res, sensitivity_frame


def fit_sarimax_candidates(frame: pd.DataFrame) -> tuple[object, pd.DataFrame, pd.DataFrame]:
    print('\n开始拟合 SARIMAX(1, 0, 1) 作为正式时间序列误差模型...')
    sensitivity_frame = pd.DataFrame({'year_month': full_index})
    sensitivity_frame['year'] = sensitivity_frame['year_month'].dt.year
    sensitivity_frame['month'] = sensitivity_frame['year_month'].dt.month
    sensitivity_frame = pd.merge(
        sensitivity_frame,
        mm_clean_original[['year_month', 'value [ng/L]', 'observed']],
        on='year_month',
        how='left',
    )
    sensitivity_frame['observed'] = sensitivity_frame['observed'].fillna(0).astype(int)
    sensitivity_frame = build_itsa_frame(sensitivity_frame)

    endog = sensitivity_frame['value [ng/L]']
    exog = sensitivity_frame[['time', 'period', 'time_after']]
    order = (1, 0, 1)
    best_res = fit_sarimax_with_retry(endog, exog, order, 'Final')

    comparison = pd.DataFrame([
        {
            'order': str(order),
            'AIC': best_res.aic,
            'BIC': best_res.bic,
            'LogLik': best_res.llf,
            'LjungBox_p_lag12': float(acorr_ljungbox(pd.Series(best_res.resid).dropna(), lags=[12], return_df=True)['lb_pvalue'].dropna().iloc[-1]),
        }
    ])

    print('\n=== SARIMAX(1, 0, 1) 拟合结果 ===')
    print(best_res.summary())

    pred = best_res.get_prediction(start=0, end=len(sensitivity_frame) - 1, exog=exog)
    sensitivity_frame['sarimax_fitted'] = pred.predicted_mean
    sarimax_ci = pred.conf_int()
    sensitivity_frame['sarimax_ci_lower'], sensitivity_frame['sarimax_ci_upper'] = ci_bounds(sarimax_ci)
    return best_res, sensitivity_frame, comparison


def sarimax_results_table(model: object, order_label: str = 'SARIMAX') -> pd.DataFrame:
    params = model.params
    pvalues = model.pvalues
    conf = model.conf_int()
    return pd.DataFrame([
        {
            'Model': f'{order_label} Sensitivity',
            'AIC': model.aic,
            'BIC': model.bic,
            'LogLik': model.llf,
            'Intercept': params.get('const', np.nan),
            'Pre_trend_time': params.get('time', np.nan),
            'Level_change_period': params.get('period', np.nan),
            'Slope_change_time_after': params.get('time_after', np.nan),
            'P_value_time': pvalues.get('time', np.nan),
            'P_value_period': pvalues.get('period', np.nan),
            'P_value_time_after': pvalues.get('time_after', np.nan),
            'CI_low_time': conf.loc['time', 0] if 'time' in conf.index else np.nan,
            'CI_high_time': conf.loc['time', 1] if 'time' in conf.index else np.nan,
            'CI_low_period': conf.loc['period', 0] if 'period' in conf.index else np.nan,
            'CI_high_period': conf.loc['period', 1] if 'period' in conf.index else np.nan,
            'CI_low_time_after': conf.loc['time_after', 0] if 'time_after' in conf.index else np.nan,
            'CI_high_time_after': conf.loc['time_after', 1] if 'time_after' in conf.index else np.nan,
        }
    ])


def save_sarimax_plot(frame: pd.DataFrame, output_name: str) -> None:
    fig, ax = plt.subplots(figsize=(16, 8))
    observed = frame[frame['value [ng/L]'].notna()]
    ax.scatter(observed['year_month'], observed['value [ng/L]'], s=36, color='black', alpha=0.85, label='Observed')
    ax.plot(frame['year_month'], frame['sarimax_fitted'], color='#1f77b4', lw=2, label='SARIMAX fitted')
    ax.fill_between(frame['year_month'], frame['sarimax_ci_lower'], frame['sarimax_ci_upper'], color='#1f77b4', alpha=0.18, label='95% CI')
    ax.axvline(intervention_date, color='crimson', ls='--', lw=1.6, label='Intervention')
    ax.set_title('SARIMAX Sensitivity Analysis', fontsize=14)
    ax.set_xlabel('Year')
    ax.set_ylabel('PFOS Concentration [ng/L]')
    xticks = frame[frame['month'] == 1]['year_month']
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(x.year) for x in xticks], rotation=45)
    ax.set_xlim(frame['year_month'].min(), frame['year_month'].max())
    ax.set_ylim(frame['value [ng/L]'].min() * 0.9, frame['value [ng/L]'].max() * 1.1)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_path = RESULTS_OUTPUT_DIR / output_name
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'SARIMAX 敏感性图已保存到：{out_path}')


def print_main_results_summary(results: pd.DataFrame) -> None:
    summary_columns = [
        'Model',
        'Pre_trend_time',
        'Level_change_period',
        'Slope_change_time_after',
        'SE_HAC_time',
        'SE_HAC_period',
        'SE_HAC_time_after',
        'P_HAC_time',
        'P_HAC_period',
        'P_HAC_time_after',
        'CI_HAC_low_period',
        'CI_HAC_high_period',
        'CI_HAC_low_time_after',
        'CI_HAC_high_time_after',
    ]
    available_columns = [col for col in summary_columns if col in results.columns]
    print('\n=== 论文主结果摘要（建议用于正文）===')
    print(results[available_columns].round(4).to_string(index=False))
    print('\n说明：正文优先使用 HAC 标准误、HAC p 值和 HAC 95% CI；普通 OLS 结果仅作参考。')


def print_core_inference_summary(ols_results: pd.DataFrame, sarimax_results: pd.DataFrame, sarimax_comparison: pd.DataFrame) -> None:
    print('\n=== 核心推断摘要 ===')
    core_rows = []
    for _, row in ols_results.iterrows():
        core_rows.append({
            'Model': row['Model'],
            'Core_beta_period': row.get('Level_change_period', np.nan),
            'Core_beta_time_after': row.get('Slope_change_time_after', np.nan),
            'HAC_p_period': row.get('P_HAC_period', np.nan),
            'HAC_p_time_after': row.get('P_HAC_time_after', np.nan),
            'HAC_CI_period': f"[{row.get('CI_HAC_low_period', np.nan):.4f}, {row.get('CI_HAC_high_period', np.nan):.4f}]" if pd.notna(row.get('CI_HAC_low_period', np.nan)) else np.nan,
            'HAC_CI_time_after': f"[{row.get('CI_HAC_low_time_after', np.nan):.4f}, {row.get('CI_HAC_high_time_after', np.nan):.4f}]" if pd.notna(row.get('CI_HAC_low_time_after', np.nan)) else np.nan,
        })
    print(pd.DataFrame(core_rows).round(4).to_string(index=False))
    print('\n=== SARIMAX 误差结构比较摘要 ===')
    print(sarimax_comparison.round(4).to_string(index=False))
    print('\n说明：主推断建议以 HAC 结果为正文分段回归结论，以 AIC 最优的 SARIMAX 作为正式时间序列误差模型的稳健性/替代推断。')


def paper_ready_parameter_table(model, cov_type: str, maxlags: int = 4) -> pd.DataFrame:
    final_res = get_final_model_results(model, cov_type=cov_type, maxlags=maxlags)
    param_index = model.params.index
    params = pd.Series(np.asarray(final_res.params), index=param_index)
    bse = pd.Series(np.asarray(final_res.bse), index=param_index)
    pvalues = pd.Series(np.asarray(final_res.pvalues), index=param_index)
    conf = pd.DataFrame(np.asarray(final_res.conf_int()), index=param_index)
    paper_rows = []
    for param_name in ['Intercept', 'time', 'period', 'time_after']:
        if param_name in params.index:
            paper_rows.append({
                'Parameter': param_name,
                'Estimate': params.get(param_name, np.nan),
                'SE': bse.get(param_name, np.nan),
                'p_value': pvalues.get(param_name, np.nan),
                'CI_low': conf.loc[param_name, 0] if param_name in conf.index else np.nan,
                'CI_high': conf.loc[param_name, 1] if param_name in conf.index else np.nan,
            })
    return pd.DataFrame(paper_rows)


def print_paper_ready_results(model, cov_type: str, maxlags: int = 4) -> None:
    paper_df = paper_ready_parameter_table(model, cov_type=cov_type, maxlags=maxlags)
    print('\n=== 论文可直接使用的最终参数结果 ===')
    print(f'最终采用的标准误类型：{cov_type}')
    print(paper_df.round(4).to_string(index=False))


main_itsa_data = build_itsa_frame(data)
ols_models = fit_ols_models(main_itsa_data)

bp_info_level = check_heteroscedasticity(ols_models['Level Change Only'], 'Level Change Only')
bp_info_slope = check_heteroscedasticity(ols_models['Slope Change Only'], 'Slope Change Only')
bp_info_final = check_heteroscedasticity(ols_models['Level + Slope Change'], 'Level + Slope Change')

dw_info_level = check_dw(ols_models['Level Change Only'], 'Level Change Only')
dw_info_slope = check_dw(ols_models['Slope Change Only'], 'Slope Change Only')
dw_info_final = check_dw(ols_models['Level + Slope Change'], 'Level + Slope Change')

plot_cov_type = {
    'Level Change Only': choose_final_cov_type(bp_info_level, dw_info_level),
    'Slope Change Only': choose_final_cov_type(bp_info_slope, dw_info_slope),
    'Level + Slope Change': choose_final_cov_type(bp_info_final, dw_info_final),
}
print('\n绘图 CI 选择（按模型诊断自动选择）：')
for model_name, ci_type in plot_cov_type.items():
    print(f'- {model_name}: {ci_type}')

final_cov_type = choose_final_cov_type(bp_info_final, dw_info_final)
print(f"\n最终论文模型的标准误选择结果：{final_cov_type}")
print_paper_ready_results(ols_models['Level + Slope Change'], final_cov_type, maxlags=4)

ols_results = ols_results_table(ols_models)
main_itsa_results_path = RESULTS_OUTPUT_DIR / 'PFOS_ITSA_Results.csv'
ols_results.to_csv(main_itsa_results_path, index=False, encoding='utf-8-sig')
print(f'\nOLS ITSA 结果已保存到：{main_itsa_results_path}')
print('\nOLS ITSA 结果预览（摘要）：')
print(ols_results[['Model', 'Level_change_period', 'Slope_change_time_after', 'P_value_period', 'P_value_time_after']].round(4).to_string(index=False))
print_main_results_summary(ols_results)

# Save the diagnostic table (including Newey-West SE) and block-bootstrap results
out_dir = RESULTS_OUTPUT_DIR
try:
    save_diagnostic_tables(ols_models, out_dir)
except Exception as e:
    print('保存诊断表失败：', e)

# Print block-bootstrap and Mann-Whitney results directly in the terminal instead of writing them to CSV
try:
    mw_summary = pd.DataFrame([
        {'metric': 'median_pre', 'value': float(median_before)},
        {'metric': 'median_post', 'value': float(median_after)},
        {'metric': 'mw_U', 'value': float(mw_stat)},
        {'metric': 'mw_p', 'value': float(mw_p)},
        {'metric': 'boot_median_diff', 'value': float(obs_diff)},
        {'metric': 'boot_p', 'value': float(p_boot)},
        {'metric': 'boot_q025', 'value': float(np.quantile(boot_dist, 0.025))},
        {'metric': 'boot_q975', 'value': float(np.quantile(boot_dist, 0.975))},
    ])
    print('\n=== Mann-Whitney / block-bootstrap 结果摘要（仅 terminal 输出）===')
    print(mw_summary.to_string(index=False))
    print('\nBootstrap 分布摘要：')
    print(pd.Series(boot_dist, name='boot_diff').describe().to_string())
except Exception as e:
    print('输出 Mann-Whitney / bootstrap 结果失败：', e)

save_itsa_plot(main_itsa_data, ols_models['Level Change Only'], 'Model 1: Level Change Only', 'ITSA_Level_Change_Only.png', ci_choice=plot_cov_type['Level Change Only'])
save_itsa_plot(main_itsa_data, ols_models['Slope Change Only'], 'Model 2: Slope Change Only', 'ITSA_Slope_Change_Only.png', ci_choice=plot_cov_type['Slope Change Only'])
save_itsa_plot(main_itsa_data, ols_models['Level + Slope Change'], 'Model 3: Level + Slope Change', 'ITSA_Level_Slope_Change.png', ci_choice=plot_cov_type['Level + Slope Change'])

# Export Model 3 fitted and counterfactual values (OLS point estimates and HAC mean CIs), diagnostics, and parametric-bootstrap counterfactual CIs
try:
    model3 = ols_models['Level + Slope Change']
    # Fitted mean and analytical HAC CI
    pred3, ci_low3, ci_high3 = compute_mean_hac_ci(model3, main_itsa_data, maxlags=4)

    # Counterfactual: set period and time_after to 0
    data_cf = main_itsa_data.copy()
    data_cf['period'] = 0
    data_cf['time_after'] = 0
    pred_cf3, ci_cf_low3, ci_cf_high3 = compute_mean_hac_ci(model3, data_cf, maxlags=4)

    # --- Diagnostic output: compare HAC mean-SE distributions for fitted and counterfactual values ---
    try:
        robust = model3.get_robustcov_results(cov_type='HAC', maxlags=4)
        V = robust.cov_params().values if hasattr(robust.cov_params(), 'values') else np.asarray(robust.cov_params())
    except Exception:
        V = model3.cov_params().values if hasattr(model3.cov_params(), 'values') else np.asarray(model3.cov_params())

    X = np.asarray(sm.add_constant(main_itsa_data[['time', 'period', 'time_after']]))
    Xcf = np.asarray(sm.add_constant(data_cf[['time', 'period', 'time_after']]))
    se = np.sqrt(np.einsum('ij,jk,ik->i', X, V, X))
    se_cf = np.sqrt(np.einsum('ij,jk,ik->i', Xcf, V, Xcf))
    print('\n=== HAC diagnostic: SE summary (mean/min/max) ===')
    print('SE (fitted):', np.nanmean(se), np.nanmin(se), np.nanmax(se))
    print('SE (cf)    :', np.nanmean(se_cf), np.nanmin(se_cf), np.nanmax(se_cf))

    # Print comparisons near the intervention point and each parameter's percentage contribution to variance
    try:
        t0_idx = int(main_itsa_data.loc[main_itsa_data['period'] == 1].index.min())
    except Exception:
        t0_idx = None
    if t0_idx is not None:
        rows = list(range(max(0, t0_idx - 3), min(len(X), t0_idx + 4)))
        print('\n=== Row-level HAC var comparison around intervention ===')
        for i in rows:
            vi = float(X[i] @ V @ X[i])
            vcf = float(Xcf[i] @ V @ Xcf[i])
            contrib = X[i] * (V @ X[i])
            contrib_pct = 100.0 * contrib / (vi if vi != 0 else np.nan)
            print(f'row {i}: se={np.sqrt(vi):.4f}, se_cf={np.sqrt(vcf):.4f}, contrib_pct(%)={np.round(contrib_pct,2)}')

    # --- Parametric bootstrap using HAC cov to get CF CI (accounts for Var(beta) estimation uncertainty) ---
    rng = np.random.default_rng(12345)
    betas = np.asarray(model3.params)
    p = betas.size
    nboot = 1000
    try:
        # try Cholesky; fallback to eigen decomposition
        try:
            L = np.linalg.cholesky(V)
        except Exception:
            vals, vecs = np.linalg.eigh((V + V.T) / 2.0)
            vals[vals < 0] = 0.0
            L = vecs @ np.diag(np.sqrt(vals))
        Z = rng.normal(size=(nboot, p))
        beta_samp = betas + Z @ L.T
        # preds: nboot x Tcf
        pred_samps_cf = beta_samp @ Xcf.T
        ci_cf_low_boot = np.quantile(pred_samps_cf, 0.025, axis=0)
        ci_cf_high_boot = np.quantile(pred_samps_cf, 0.975, axis=0)
    except Exception as e:
        print('Param-bootstrap failed:', e)
        ci_cf_low_boot = ci_cf_low3
        ci_cf_high_boot = ci_cf_high3

    # Build the output table with analytical HAC CIs and parametric-bootstrap counterfactual CIs
    fitted_data = pd.DataFrame({
        'year_month': main_itsa_data['year_month'],
        'Model 3 Fitted': np.asarray(pred3),
        'Model 3 CI Lower (HAC)': np.asarray(ci_low3),
        'Model 3 CI Upper (HAC)': np.asarray(ci_high3),
        'Model 3 Counterfactual': np.asarray(pred_cf3),
        'Model 3 CF CI Lower (HAC)': np.asarray(ci_cf_low3),
        'Model 3 CF CI Upper (HAC)': np.asarray(ci_cf_high3),
        'Model 3 CF CI Lower (Boot)': np.asarray(ci_cf_low_boot),
        'Model 3 CF CI Upper (Boot)': np.asarray(ci_cf_high_boot),
    })

    out_excel = RESULTS_OUTPUT_DIR / 'fitted_data.xlsx'
    fitted_data.to_excel(out_excel, index=False)
    print(f'已导出 Model 3 拟合与反事实数据到：{out_excel} (包含 HAC analytic CI 与 param-bootstrap CF CI)')
except Exception as e:
    print('导出 Model 3 拟合/反事实 数据失败：', e)

sarimax_res, sarimax_frame, sarimax_comparison = fit_sarimax_candidates(data)
sarimax_results = sarimax_results_table(sarimax_res, order_label=f'SARIMAX{sarimax_res.model.order}')
sarimax_results_path = RESULTS_OUTPUT_DIR / 'PFOS_ITSA_SARIMAX_Sensitivity.csv'
sarimax_results.to_csv(sarimax_results_path, index=False, encoding='utf-8-sig')
print(f'\nSARIMAX 敏感性结果已保存到：{sarimax_results_path}')
print('\nSARIMAX 敏感性结果预览：')
print(sarimax_results.to_string(index=False))

sarimax_comparison_path = RESULTS_OUTPUT_DIR / 'PFOS_SARIMAX_Order_Comparison.csv'
sarimax_comparison.to_csv(sarimax_comparison_path, index=False, encoding='utf-8-sig')
print(f'\nSARIMAX 误差结构比较结果已保存到：{sarimax_comparison_path}')

print_core_inference_summary(ols_results, sarimax_results, sarimax_comparison)

sarimax_fit_path = RESULTS_OUTPUT_DIR / 'sarimax_sensitivity_fitted.csv'
sarimax_frame.to_csv(sarimax_fit_path, index=False, encoding='utf-8-sig')
print(f'\nSARIMAX 拟合序列已保存到：{sarimax_fit_path}')
save_sarimax_plot(sarimax_frame, 'SARIMAX_Sensitivity.png')


def run_full_analysis_for_csv(input_csv: str, output_dir: Path, label: str) -> None:
    global RESULTS_OUTPUT_DIR, df1, median_raw, median_cell, median_ref, median_year_month
    global mm, mm_original, model_original, observed_mask, residuals_observed, std_residuals_observed
    global outliers_original, mm_clean_original, complete_ts, data, group_before, group_after
    global median_before, median_after, mw_stat, mw_p, obs_series, obs_dates, obs_diff, p_boot, boot_dist
    global main_itsa_data, ols_models, ols_results, sarimax_res, sarimax_frame, sarimax_comparison

    RESULTS_OUTPUT_DIR = output_dir
    RESULTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f'\n========== 开始重新运行：{label} ==========' )
    df1 = pd.read_csv(
        input_csv,
        usecols=['year', 'month', 'Substance', 'lon', 'lat', 'lon_grid', 'lat_grid', 'country', 'value [ng/L]', 'source', 'limit [ng/L]'],
        dtype={'value [ng/L]': float},
    )

    df1['source'] = df1['source'].fillna('Unknown')
    df1['limit [ng/L]'] = df1['limit [ng/L]'].fillna(0.003536)
    df1.loc[df1['value [ng/L]'] == 0, 'value [ng/L]'] = np.nan
    df1 = df1[~((df1['limit [ng/L]'] >= 1) & (df1['value [ng/L]'].isna()))].copy()
    missing_value_mask = df1['value [ng/L]'].isna()
    df1.loc[missing_value_mask, 'value [ng/L]'] = df1.loc[missing_value_mask, 'limit [ng/L]'] / np.sqrt(2)

    median_raw = df1.groupby(['year', 'month', 'source', 'lon', 'lat', 'lon_grid', 'lat_grid'])['value [ng/L]'].median().reset_index()
    median_cell = median_raw.groupby(['year', 'month', 'source', 'lon_grid', 'lat_grid'])['value [ng/L]'].median().reset_index()
    median_ref = median_cell.groupby(['year', 'month', 'lon_grid', 'lat_grid'])['value [ng/L]'].median().reset_index()
    median_year_month = median_ref.groupby(['year', 'month'])['value [ng/L]'].median().reset_index()
    median_ref.to_excel(RESULTS_OUTPUT_DIR / f'{label}_global_monthly_cell_PFOS.xlsx', index=False)
    median_year_month.to_excel(RESULTS_OUTPUT_DIR / f'{label}_global_monthly_median_PFOS.xlsx', index=False)

    mm = median_year_month.copy()
    mm['year'] = mm['year'].astype(int)
    mm['month'] = mm['month'].astype(int)
    mm['year_month'] = pd.to_datetime(mm['year'].astype(str) + '-' + mm['month'].astype(str).str.zfill(2), format='%Y-%m')
    mm = mm.sort_values('year_month').drop_duplicates(subset=['year_month'])
    full_index = pd.date_range(start=mm['year_month'].min(), end=mm['year_month'].max(), freq='MS')

    print('基于原始观测数据进行异常值检测...')
    mm_original = mm.copy()
    mm_original['time'] = mm_original['year'] + (mm_original['month'] / 12.0)
    observed_mask = ~mm_original['value [ng/L]'].isna()
    X_observed = sm.add_constant(mm_original.loc[observed_mask, 'time'])
    y_observed = mm_original.loc[observed_mask, 'value [ng/L]']
    model_original = sm.OLS(y_observed, X_observed).fit()
    mm_original['residuals'] = np.nan
    mm_original.loc[observed_mask, 'residuals'] = model_original.resid
    residuals_observed = mm_original.loc[observed_mask, 'residuals']
    std_residuals_observed = zscore(residuals_observed)
    mm_original.loc[observed_mask, 'std_residuals'] = std_residuals_observed
    mm_original['is_outlier'] = False
    mm_original.loc[observed_mask, 'is_outlier'] = mm_original.loc[observed_mask, 'std_residuals'].abs() > 3
    outliers_original = mm_original.loc[mm_original['is_outlier'], ['year', 'month', 'year_month', 'value [ng/L]', 'residuals', 'std_residuals']]
    print('基于原始观测数据检测到的异常值：')
    print(outliers_original)
    outliers_original.to_excel(RESULTS_OUTPUT_DIR / f'{label}_global_PFOS_outliers_before_interp.xlsx', index=False)

    print('\n删除异常值并进行一次性线性插值...')
    mm_clean_original = mm_original[~mm_original['is_outlier']].copy()
    mm_clean_original['observed'] = 1
    complete_ts = pd.DataFrame({'year_month': full_index})
    complete_ts['year'] = complete_ts['year_month'].dt.year
    complete_ts['month'] = complete_ts['year_month'].dt.month
    complete_ts = pd.merge(complete_ts, mm_clean_original[['year_month', 'value [ng/L]', 'observed']], on='year_month', how='left')
    complete_ts['observed'] = complete_ts['observed'].fillna(0).astype(int)
    print(f"插值前缺失值数量: {complete_ts['value [ng/L]'].isna().sum()}")
    complete_ts['value [ng/L]'] = complete_ts['value [ng/L]'].interpolate(method='linear')
    if complete_ts['value [ng/L]'].isna().any():
        print('序列开头或结尾仍有缺失值，使用前向/后向填充...')
        complete_ts['value [ng/L]'] = complete_ts['value [ng/L]'].fillna(method='bfill').fillna(method='ffill')
    data = complete_ts.copy()

    median_year_month['date'] = pd.to_datetime(median_year_month['year'].astype(str) + '-' + median_year_month['month'].astype(str).str.zfill(2) + '-01')
    cutoff_date = pd.to_datetime('2010-08-01')
    group_before = median_year_month.loc[median_year_month['date'] < cutoff_date, 'value [ng/L]']
    group_after = median_year_month.loc[median_year_month['date'] >= cutoff_date, 'value [ng/L]']
    median_before = group_before.median()
    median_after = group_after.median()
    mw_stat, mw_p = mannwhitneyu(group_before.dropna(), group_after.dropna(), alternative='two-sided')
    print('=== Mann–Whitney U 检验（传统） ===')
    print(f'2010年8月前 中位数: {median_before:.3f} ng/L, 样本量: {len(group_before.dropna())}')
    print(f'2010年8月后 中位数: {median_after:.3f} ng/L, 样本量: {len(group_after.dropna())}')
    print(f'U 统计量: {mw_stat:.3f}, p 值: {mw_p:.5f}')

    def block_bootstrap_median_diff(series: pd.Series, dates: pd.Series, cutoff: pd.Timestamp, n_boot=1000, block_size=3, seed=42):
        rng = np.random.default_rng(seed)
        series = series.reset_index(drop=True)
        dates = dates.reset_index(drop=True)
        n = len(series)
        starts = np.arange(0, n - block_size + 1)
        pre_idx = dates < cutoff
        post_idx = ~pre_idx
        obs_pre_med = np.nanmedian(series[pre_idx.values])
        obs_post_med = np.nanmedian(series[post_idx.values])
        obs_diff = obs_post_med - obs_pre_med
        boot_diffs = []
        for _ in range(n_boot):
            res = []
            while len(res) < n:
                s = int(rng.choice(starts))
                res.extend(series[s:s+block_size].tolist())
            res = np.array(res[:n])
            boot_pre_med = np.nanmedian(res[pre_idx.values])
            boot_post_med = np.nanmedian(res[post_idx.values])
            boot_diffs.append(boot_post_med - boot_pre_med)
        boot_diffs = np.array(boot_diffs)
        p_boot = np.mean(np.abs(boot_diffs) >= np.abs(obs_diff))
        return obs_diff, p_boot, boot_diffs

    obs_series = data['value [ng/L]'].copy()
    obs_dates = data['year_month'].copy()
    obs_diff, p_boot, boot_dist = block_bootstrap_median_diff(obs_series, obs_dates, cutoff_date, n_boot=1000, block_size=3)
    print('\n=== Block-bootstrap (median diff) ===')
    print(f'Intervention 后-前 中位数差: {obs_diff:.5f} ng/L, bootstrap p-value: {p_boot:.4f}')

    def safe_adf(series: pd.Series, name: str):
        s = series.dropna()
        print(f"\n【ADF 检验 - {name}】 样本量: {len(s)}")
        if len(s) < 10:
            print('样本过少，ADF 结果不可靠（建议至少 ~10 个观测点）。')
            return None
        try:
            res = adfuller(s)
            print(f'ADF 统计量: {res[0]:.4f}, p 值: {res[1]:.4f}')
            print('序列为' + ('非平稳' if res[1] > 0.05 else '平稳'))
            return res
        except Exception as e:
            print('ADF 检验出错:', e)
            return None

    safe_adf(data.loc[data['year_month'] < cutoff_date, 'value [ng/L]'], 'Interpolated - Pre')
    safe_adf(data.loc[data['year_month'] >= cutoff_date, 'value [ng/L]'], 'Interpolated - Post')
    safe_adf(median_year_month.loc[median_year_month['date'] < cutoff_date, 'value [ng/L]'], 'Observed aggregated - Pre')
    safe_adf(median_year_month.loc[median_year_month['date'] >= cutoff_date, 'value [ng/L]'], 'Observed aggregated - Post')

    def check_heteroscedasticity_local(model, name):
        lm, lm_pvalue, fvalue, f_pvalue = het_breuschpagan(model.resid, model.model.exog)
        print(f"\n【Breusch-Pagan 异方差检验 - {name}】")
        print(f"LM统计量: {lm:.4f}, p值: {lm_pvalue:.4f}")
        print(f"F统计量: {fvalue:.4f}, p值: {f_pvalue:.4f}")
        has_heteroscedasticity = lm_pvalue < 0.05
        return {'lm_pvalue': lm_pvalue, 'f_pvalue': f_pvalue, 'has_heteroscedasticity': has_heteroscedasticity}

    def check_dw_local(model, name):
        dw = durbin_watson(model.resid)
        print(f"\n【Durbin-Watson 检验 - {name}】")
        print(f"DW 值: {dw:.4f}")
        has_autocorrelation = dw < 1.5 or dw > 2.5
        try:
            bg = acorr_breusch_godfrey(model, nlags=4)
            bg_pvalue = bg[1]
            has_autocorrelation = has_autocorrelation or (bg_pvalue < 0.05)
        except Exception:
            bg_pvalue = np.nan
        try:
            lb = acorr_ljungbox(model.resid, lags=[12], return_df=True)
            lb_pvalue = float(lb['lb_pvalue'].iloc[-1])
            has_autocorrelation = has_autocorrelation or (lb_pvalue < 0.05)
        except Exception:
            lb_pvalue = np.nan
        return {'dw': dw, 'bg_pvalue': bg_pvalue, 'lb_pvalue': lb_pvalue, 'has_autocorrelation': has_autocorrelation}

    main_itsa_data = build_itsa_frame(data)
    ols_models = fit_ols_models(main_itsa_data)
    bp_info_level = check_heteroscedasticity_local(ols_models['Level Change Only'], 'Level Change Only')
    bp_info_slope = check_heteroscedasticity_local(ols_models['Slope Change Only'], 'Slope Change Only')
    bp_info_final = check_heteroscedasticity_local(ols_models['Level + Slope Change'], 'Level + Slope Change')
    dw_info_level = check_dw_local(ols_models['Level Change Only'], 'Level Change Only')
    dw_info_slope = check_dw_local(ols_models['Slope Change Only'], 'Slope Change Only')
    dw_info_final = check_dw_local(ols_models['Level + Slope Change'], 'Level + Slope Change')
    plot_cov_type = {
        'Level Change Only': choose_final_cov_type(bp_info_level, dw_info_level),
        'Slope Change Only': choose_final_cov_type(bp_info_slope, dw_info_slope),
        'Level + Slope Change': choose_final_cov_type(bp_info_final, dw_info_final),
    }
    print('\n绘图 CI 选择（按模型诊断自动选择）：')
    for model_name, ci_type in plot_cov_type.items():
        print(f'- {model_name}: {ci_type}')
    final_cov_type = choose_final_cov_type(bp_info_final, dw_info_final)
    print(f"\n最终论文模型的标准误选择结果：{final_cov_type}")
    print_paper_ready_results(ols_models['Level + Slope Change'], final_cov_type, maxlags=4)

    ols_results = ols_results_table(ols_models)
    main_itsa_results_path = RESULTS_OUTPUT_DIR / 'PFOS_ITSA_Results.csv'
    ols_results.to_csv(main_itsa_results_path, index=False, encoding='utf-8-sig')
    print(f'\nOLS ITSA 结果已保存到：{main_itsa_results_path}')
    print(ols_results[['Model', 'Level_change_period', 'Slope_change_time_after', 'P_value_period', 'P_value_time_after']].round(4).to_string(index=False))

    try:
        save_diagnostic_tables(ols_models, RESULTS_OUTPUT_DIR)
    except Exception as e:
        print('保存诊断表失败：', e)

    save_itsa_plot(main_itsa_data, ols_models['Level Change Only'], 'Model 1: Level Change Only', 'ITSA_Level_Change_Only.png', ci_choice=plot_cov_type['Level Change Only'])
    save_itsa_plot(main_itsa_data, ols_models['Slope Change Only'], 'Model 2: Slope Change Only', 'ITSA_Slope_Change_Only.png', ci_choice=plot_cov_type['Slope Change Only'])
    save_itsa_plot(main_itsa_data, ols_models['Level + Slope Change'], 'Model 3: Level + Slope Change', 'ITSA_Level_Slope_Change.png', ci_choice=plot_cov_type['Level + Slope Change'])

    try:
        model3 = ols_models['Level + Slope Change']
        pred3, ci_low3, ci_high3 = compute_mean_hac_ci(model3, main_itsa_data, maxlags=4)
        data_cf = main_itsa_data.copy()
        data_cf['period'] = 0
        data_cf['time_after'] = 0
        pred_cf3, ci_cf_low3, ci_cf_high3 = compute_mean_hac_ci(model3, data_cf, maxlags=4)
        try:
            robust = model3.get_robustcov_results(cov_type='HAC', maxlags=4)
            V = robust.cov_params().values if hasattr(robust.cov_params(), 'values') else np.asarray(robust.cov_params())
        except Exception:
            V = model3.cov_params().values if hasattr(model3.cov_params(), 'values') else np.asarray(model3.cov_params())
        X = np.asarray(sm.add_constant(main_itsa_data[['time', 'period', 'time_after']]))
        Xcf = np.asarray(sm.add_constant(data_cf[['time', 'period', 'time_after']]))
        rng = np.random.default_rng(12345)
        betas = np.asarray(model3.params)
        p = betas.size
        nboot = 1000
        try:
            try:
                L = np.linalg.cholesky(V)
            except Exception:
                vals, vecs = np.linalg.eigh((V + V.T) / 2.0)
                vals[vals < 0] = 0.0
                L = vecs @ np.diag(np.sqrt(vals))
            Z = rng.normal(size=(nboot, p))
            beta_samp = betas + Z @ L.T
            pred_samps_cf = beta_samp @ Xcf.T
            ci_cf_low_boot = np.quantile(pred_samps_cf, 0.025, axis=0)
            ci_cf_high_boot = np.quantile(pred_samps_cf, 0.975, axis=0)
        except Exception:
            ci_cf_low_boot = ci_cf_low3
            ci_cf_high_boot = ci_high3
        fitted_data = pd.DataFrame({
            'year_month': main_itsa_data['year_month'],
            'Model 3 Fitted': np.asarray(pred3),
            'Model 3 CI Lower (HAC)': np.asarray(ci_low3),
            'Model 3 CI Upper (HAC)': np.asarray(ci_high3),
            'Model 3 Counterfactual': np.asarray(pred_cf3),
            'Model 3 CF CI Lower (HAC)': np.asarray(ci_cf_low3),
            'Model 3 CF CI Upper (HAC)': np.asarray(ci_cf_high3),
            'Model 3 CF CI Lower (Boot)': np.asarray(ci_cf_low_boot),
            'Model 3 CF CI Upper (Boot)': np.asarray(ci_cf_high_boot),
        })
        out_excel = RESULTS_OUTPUT_DIR / 'fitted_data.xlsx'
        fitted_data.to_excel(out_excel, index=False)
        print(f'已导出 Model 3 拟合与反事实数据到：{out_excel}')
    except Exception as e:
        print('导出 Model 3 拟合/反事实 数据失败：', e)

    sarimax_res, sarimax_frame, sarimax_comparison = fit_sarimax_candidates(data)
    sarimax_results = sarimax_results_table(sarimax_res, order_label=f'SARIMAX{sarimax_res.model.order}')
    sarimax_results_path = RESULTS_OUTPUT_DIR / 'PFOS_ITSA_SARIMAX_Sensitivity.csv'
    sarimax_results.to_csv(sarimax_results_path, index=False, encoding='utf-8-sig')
    sarimax_comparison_path = RESULTS_OUTPUT_DIR / 'PFOS_SARIMAX_Order_Comparison.csv'
    sarimax_comparison.to_csv(sarimax_comparison_path, index=False, encoding='utf-8-sig')
    sarimax_fit_path = RESULTS_OUTPUT_DIR / 'sarimax_sensitivity_fitted.csv'
    sarimax_frame.to_csv(sarimax_fit_path, index=False, encoding='utf-8-sig')
    save_sarimax_plot(sarimax_frame, 'SARIMAX_Sensitivity.png')
    print_core_inference_summary(ols_results, sarimax_results, sarimax_comparison)
    print(f'\n[{label}] 所有结果已输出到：{RESULTS_OUTPUT_DIR}')


run_full_analysis_for_csv(
    r'D:\检测小队\林嘉意\中断时间序列\Global\data\PFOS_NAH_NP.csv',
    Path(r'D:\检测小队\林嘉意\中断时间序列\Global\Part. 2\results output\No Point Source'),
    'No_Point_Source',
)

run_full_analysis_for_csv(
    r'D:\检测小队\林嘉意\中断时间序列\Global\data\PFOS_sensitivity.csv',
    Path(r'D:\检测小队\林嘉意\中断时间序列\Global\Part. 2\results output\Sensitivity analysis'),
    'Sensitivity_analysis',
)
