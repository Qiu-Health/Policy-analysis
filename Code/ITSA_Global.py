import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import zscore

df = pd.read_csv('D:\global_monthly_cell_PFOS.csv',
                    usecols=['year', 'month', 'lon_grid', 'lat_grid', 'value [ng/L]'],
                    dtype={'value [ng/L]': float})
df = df.dropna(subset=['value [ng/L]'])
median_year_month = df.groupby(['year', 'month'])[
    'value [ng/L]'].median().reset_index()

mm = median_year_month.copy()
mm['year'] = mm['year'].astype(int)
mm['month'] = mm['month'].astype(int)

mm['year_month'] = pd.to_datetime(
    mm['year'].astype(str) + '-' + mm['month'].astype(str).str.zfill(2),
    format='%Y-%m'
)
mm = mm.sort_values('year_month').drop_duplicates(subset=['year_month'])

full_index = pd.date_range(start=mm['year_month'].min(),
                           end=mm['year_month'].max(),
                           freq='MS')

print("Outlier detection...")

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
print("Outliers")
print(outliers_original)

mm_clean_original = mm_original[~mm_original['is_outlier']].copy()
mm_clean_original['observed'] = 1

complete_ts = pd.DataFrame({'year_month': full_index})
complete_ts['year'] = complete_ts['year_month'].dt.year
complete_ts['month'] = complete_ts['year_month'].dt.month

complete_ts = pd.merge(complete_ts,
                      mm_clean_original[['year_month', 'value [ng/L]', 'observed']],
                      on='year_month', how='left')

complete_ts['observed'] = complete_ts['observed'].fillna(0).astype(int)
complete_ts['value [ng/L]'] = complete_ts['value [ng/L]'].interpolate(method='linear')
if complete_ts['value [ng/L]'].isna().any():
    complete_ts['value [ng/L]'] = complete_ts['value [ng/L]'].fillna(method='bfill').fillna(method='ffill')
data = complete_ts.copy()
interpolated_points = data[data['observed'] == 0]

data['time'] = np.arange(len(data))
data['period'] = (data['year_month'] >= pd.to_datetime('2010-8')).astype(int)
T0 = data[data['year_month'] == pd.to_datetime('2010-8')]['time'].values[0]

data['time_c'] = data['time'] - T0


ITSA = sm.OLS.from_formula('Q("value [ng/L]") ~ time + period + I(time-{0}):period'.format(T0), data=data).fit()

def print_model_results(model, name):
    print(f"\n=== {name} ===")
    print("AIC:", model.aic)
    print(model.summary())
    print("Confidence Intervals:\n", model.conf_int())

print_model_results(ITSA, "Level + Slope Change")


data_counterfactual = data.copy()
data_counterfactual['period'] = 0
pred = ITSA.predict(data)
pred_cf = ITSA.predict(data_counterfactual)


def plot_model_with_ci(data, model, pred, pred_cf, title, intervention_year=2010, intervention_month=8):
    plt.figure(figsize=(16, 8))
    ax = plt.gca()

    intervention_idx = data[(data['year'] == intervention_year) & (data['month'] == intervention_month)].index[0]

    data['year_month'] = pd.to_datetime(data['year'].astype(str) + '-' + data['month'].astype(str).str.zfill(2))
    plt.scatter(data['year_month'], data['value [ng/L]'], s=60, color='grey', label='Original Data', alpha=0.4)

    observed_data = data[data["observed"] == 1]
    ax.scatter(observed_data['year_month'], observed_data['value [ng/L]'], s=100, color='black', label='Median', alpha=1, edgecolors='black', linewidths=0.5, zorder=3)

    ci = model.get_prediction(data).conf_int()
    lower, upper = ci[:, 0], ci[:, 1]
    plt.plot(data['year_month'], pred, 'b-', lw=2, label='Fitted Trend')
    plt.fill_between(data['year_month'], lower, upper, color='blue', alpha=0.2, label='95% CI (Fitted)')

    data_cf = data.copy()
    data_cf['period'] = 0
    ci_cf = model.get_prediction(data_cf).conf_int()
    lower_cf, upper_cf = ci_cf[:, 0], ci_cf[:, 1]

    plt.plot(data['year_month'][intervention_idx:], pred_cf[intervention_idx:], 'r--', lw=2, label='Counterfactual (No Intervention)')
    plt.fill_between(data['year_month'][intervention_idx:], lower_cf[intervention_idx:], upper_cf[intervention_idx:], color='red', alpha=0.2, label='95% CI (Counterfactual)')

    plt.axvline(x=data['year_month'][intervention_idx], color='k', linestyle='--', alpha=0.6, label='Intervention (Aug 2010)')

    plt.title(title + '\nGlobal PFOS Concentration Analysis', fontsize=14)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('PFOS Concentration [ng/L]', fontsize=12)

    xticks = data[(data['year'] >= 2000) & (data['year'] <= 2025) & (data['month'] == 1)]['year_month']
    plt.xticks(xticks, xticks.dt.year, rotation=45)

    plt.xlim(data['year_month'].min() - pd.DateOffset(years=1), data['year_month'].max() + pd.DateOffset(years=1))
    plt.ylim(data['value [ng/L]'].min() * 0.9, data['value [ng/L]'].max() * 1.1)

    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    plt.savefig(f"{title}_PFOS_median_plot.pdf", format="pdf", dpi=300)

plot_model_with_ci(data, ITSA, pred, pred_cf, "ITSA: Level + Slope Change")
