import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import zscore
import matplotlib
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['axes.unicode_minus'] = False

# 1. Data preprocessing
# China: D:\CHINA_median_cell_PFOS.csv
# The United States: D:\UNITED STATES_median_cell_PFOS.csv
# The European Union: D:\EU_median_cell_PFOS.csv
df1 = pd.read_csv('D:\CHINA_median_cell_PFOS.csv',
                    usecols=['year', 'month', 'lon_grid', 'lat_grid', 'value [ng/L]'],
                    dtype={'value [ng/L]': float})
df1 = df1.dropna(subset=['value [ng/L]'])
df1['semester'] = df1['month'].apply(lambda x: 'H1' if x <= 6 else 'H2')
median_month = df1.groupby(['year', 'semester', 'month'])['value [ng/L]'].median().reset_index()
median_year_semester = median_month.groupby(['year', 'semester'])['value [ng/L]'].median().reset_index()

df = median_year_semester.copy()
df['year_semester'] = (
    df['year'].astype(str) + '-' +
    df['semester'].replace({'H1': '01-01', 'H2': '07-01'})
)
df['year_semester'] = pd.to_datetime(df['year_semester'])
df = df.sort_values('year_semester').drop_duplicates('year_semester')

df['time'] = np.arange(len(df))

obs = ~df['value [ng/L]'].isna()
X = sm.add_constant(df.loc[obs, 'time'])
y = df.loc[obs, 'value [ng/L]']

model = sm.OLS(y, X).fit()
df.loc[obs, 'resid'] = model.resid
df.loc[obs, 'z'] = zscore(df.loc[obs, 'resid'])

df['is_outlier'] = False
df.loc[obs, 'is_outlier'] = df.loc[obs, 'z'].abs() > 3

outliers = df.loc[df['is_outlier'], ['year', 'semester', 'value [ng/L]', 'resid', 'z']]
print('Outliers：')
print(outliers)
outliers.to_csv('semester_outliers.csv', index=False, encoding='utf-8-sig')

clean_obs = df[~df['is_outlier']].copy()
clean_obs['observed'] = 1

full_idx = pd.date_range(start=df['year_semester'].min(),
                         end=df['year_semester'].max(),
                         freq='6MS')

data = pd.DataFrame({'year_semester': full_idx})
data = data.merge(clean_obs[['year_semester', 'value [ng/L]', 'observed']],
                  on='year_semester', how='left')

data['value [ng/L]'] = data['value [ng/L]'].interpolate(method='linear')

data['value [ng/L]'] = data['value [ng/L]'].bfill().ffill()

data['year'] = data['year_semester'].dt.year
data['month'] = data['year_semester'].dt.month
data['year_month'] = data['year_semester']
data = data.reset_index(drop=True)

# 2. ITSA, intervention points: '2014-01-01' for China; '2007-10-01' for The United States; '2010-8-01' for The European Union
data['time'] = np.arange(len(data))
data['period'] = (data['year_month'] >= pd.to_datetime('2014-01-01')).astype(int)
T0 = data[data['year_month'] == pd.to_datetime('2014-01-01')]['time'].values[0]
data['time_c'] = data['time'] - T0
ITSA = sm.OLS.from_formula('Q("value [ng/L]") ~ time + period + I(time-{0}):period'.format(T0), data=data).fit()


def print_model_results(model, name):
    print(f"\n=== {name} ===")
    print("AIC:", model.aic)
    print(model.summary())
    print("Confidence Intervals:\n", model.conf_int())

print_model_results(ITSA, "Level + Slope Change")

def get_predictions(model, data_input):
    pred = model.predict(data_input)
    data_cf = data_input.copy()
    data_cf['period'] = 0
    pred_cf = model.predict(data_cf)
    return pred, pred_cf

pred, pred_cf = get_predictions(ITSA, data)

# 3. Drawing function
def plot_model_with_ci(data, model, pred, pred_cf, title, intervention_year=2014, intervention_month=1):
    plt.figure(figsize=(12, 6))
    ax = plt.gca()

    intervention_idx = data[(data['year'] == intervention_year) & (data['month'] == intervention_month)].index[0]

    ax.add_patch(Rectangle((intervention_idx, 0),
                           len(data) - intervention_idx,
                           data['value [ng/L]'].max() * 1.2,
                           color='grey', alpha=0.2))

    observed_data = data[data["observed"] == 1]
    plt.scatter(observed_data.index, observed_data['value [ng/L]'], s=30, label='Observed PFOS Concentration')

    ci = model.get_prediction(data).conf_int()
    lower, upper = ci[:, 0], ci[:, 1]
    plt.plot(data.index, pred, 'b-', lw=2, label='Fitted Trend')
    plt.fill_between(data.index, lower, upper, color='blue', alpha=0.2, label='95% CI (Fitted)')

    data_cf = data.copy()
    data_cf['period'] = 0
    ci_cf = model.get_prediction(data_cf).conf_int()
    lower_cf, upper_cf = ci_cf[:, 0], ci_cf[:, 1]
    plt.plot(data.index, pred_cf, 'r--', lw=2, label='Counterfactual (No Intervention)')
    plt.fill_between(data.index, lower_cf, upper_cf, color='red', alpha=0.2, label='95% CI (Counterfactual)')

    plt.axvline(x=intervention_idx, color='k', linestyle='--', alpha=0.6, label='Intervention (2014)')

    plt.title(title + '\nGlobal PFOS Concentration Analysis', fontsize=14)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('PFOS Concentration [ng/L]', fontsize=12)

    xticks = data[data['month'] == 1].index
    plt.xticks(xticks, data.loc[xticks, 'year'], rotation=45)

    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

plot_model_with_ci(data, ITSA, pred, pred_cf, "Level + Slope Change")

