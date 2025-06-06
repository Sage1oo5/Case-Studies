import os, re
import numpy as np
import pandas as pd
from statsmodels.multivariate.manova import MANOVA
import patsy
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX

# 1) read and combine all the 33 raw datasets
file_path = '/Users/mandyking/Documents/Germany/Study/Semester3/case study/OHNE PV'
csvs = sorted([f for f in os.listdir(file_path) if f.endswith('.csv')])
frames = []
for idx, fn in enumerate(csvs, start=1):
    tmp = pd.read_csv(os.path.join(file_path, fn), parse_dates=['index'])
    # date/time
    tmp = tmp[['index','HAUSHALT_TOT']]
    tmp['date'] = tmp['index'].dt.date
    tmp['time'] = tmp['index'].dt.strftime('%H:%M')
    # date×time
    pv = tmp.pivot(index='date', columns='time', values='HAUSHALT_TOT').sort_index(axis=1)
    pv = pv.reset_index()  # 保留 date
    pv['factor_household'] = idx
    pv['factor_day']     = pd.to_datetime(pv['date']).dt.dayofyear
    pv['factor_weekday'] = pd.to_datetime(pv['date']).dt.weekday + 1
    pv['season']         = np.sin((pv['factor_day']-52)*2*np.pi/365)
    frames.append(pv)
df = pd.concat(frames, ignore_index=True)

# 1a) missing value and smooth
time96 = [c for c in df.columns if re.fullmatch(r'\d{1,2}:\d{2}', c)]
# linear interpolation
df[time96] = df[time96].interpolate(axis=1, limit_direction='both')
df[time96] = df[time96].fillna(df[time96].mean(axis=1), axis=0)
# Method a: 5‐point moving average
df_smooth_a = df.copy()
df_smooth_a[time96] = df_smooth_a[time96].T.rolling(window=5, min_periods=1, center=True).mean().T
#  Method b: 5‐point moving average of the same weekday
window_days = 7
df_smooth_b = df.copy()
df_smooth_b[time96] = (
    df_smooth_b.groupby('factor_weekday')[time96]
    .apply(lambda x: x.rolling(window=window_days, min_periods=1, center=True).mean())
    .reset_index(drop=True)
)
# —————————————————————————————————————————————————————————————————————————————
# 2)rename 96 columns-> cluster to 24 hours
rename_map = {c:f"t{int(c.split(':')[0]):02d}_{int(c.split(':')[1]):02d}" for c in time96}
df_smooth_a = df_smooth_a.rename(columns=rename_map)
df_smooth_b = df_smooth_b.rename(columns=rename_map)
df=df.rename(columns=rename_map)

# clustering to 24 hours
for h in range(24):
    cols = [f"t{h:02d}_{m:02d}" for m in (0,15,30,45)]
    df_smooth_a[f"hour_{h}"] = df_smooth_a[cols].mean(axis=1)
hour_cols = [f"hour_{h}" for h in range(24)]

for h in range(24):
    cols = [f"t{h:02d}_{m:02d}" for m in (0,15,30,45)]
    df_smooth_b[f"hour_{h}"] = df_smooth_b[cols].mean(axis=1)
hour_cols = [f"hour_{h}" for h in range(24)]
for h in range(24):
    cols = [f"t{h:02d}_{m:02d}" for m in (0,15,30,45)]
    df[f"hour_{h}"] = df[cols].mean(axis=1)
hour_cols = [f"hour_{h}" for h in range(24)]

df_smooth_a['is_weekend'] = (df_smooth_a['factor_weekday']>=6).astype(int)
df_smooth_b['is_weekend'] = (df_smooth_b['factor_weekday']>=6).astype(int)
df['is_weekend'] = (df['factor_weekday']>=6).astype(int)

#MANOVA check before normalization
from scipy.stats import chi2
from itertools import combinations

Y = df_smooth_a[hour_cols].values
labels_house = df_smooth_a['factor_household'].values
labels_week  = df_smooth_a['is_weekend'].values

N, p = Y.shape
groups = np.unique(labels_house)

# ——————————————————————————————
# Mardia’s
def mardia_test(Y):
    N, p = Y.shape
    mean_vec = np.mean(Y, axis=0)
    S = np.cov(Y, rowvar=False)
    invS = np.linalg.inv(S)
    Z = Y - mean_vec  # N×p
    g1 = 0.0
    for i in range(N):
        Zi = Z[i]
        for j in range(N):
            Zj = Z[j]
            val = float(Zi.dot(invS).dot(Zj))
            g1 += val**3
    g1 /= (N**2)
    di = np.sum((Z.dot(invS)) * Z, axis=1)  # (z_i^T invS z_i) for each i
    g2 = np.mean(di**2)

    return g1, g2

g1, g2 = mardia_test(Y)
print(f"Mardia’s skewness (g1) = {g1:.3f}, kurtosis (g2) = {g2:.3f}")
df_skew = p*(p+1)*(p+2)/6
chi2_skew = N * g1 / 6.0
pval_skew = 1 - chi2.cdf(chi2_skew, df_skew)
print(f" -> Mardia’s skewness χ² ≈ {chi2_skew:.1f}, df={df_skew}, p={pval_skew:.3f}")

expected_kurt = p*(p+2)
std_kurt = np.sqrt( 8*p*(p+2)/N )
z_kurt = (g2 - expected_kurt) / std_kurt
from scipy.stats import norm
pval_kurt = 2*(1 - norm.cdf(abs(z_kurt)))
print(f" -> Mardia’s kurtosis z ≈ {z_kurt:.3f}, p={pval_kurt:.3f}")
# ——————————————————————————————
# Box’s M
def box_m_test(Y, labels):
    unique_labels = np.unique(labels)
    g = len(unique_labels)
    n_i = []
    cov_i = []
    for lab in unique_labels:
        Yi = Y[labels == lab]
        n_i.append(Yi.shape[0])
        cov_i.append(np.cov(Yi, rowvar=False))
    n_i = np.array(n_i)
    N = np.sum(n_i)
    p = Y.shape[1]
    Spooled = np.zeros((p, p))
    for ni, Si in zip(n_i, cov_i):
        Spooled += (ni - 1) * Si
    Spooled = Spooled / (N - g)
    det_sp = np.linalg.det(Spooled)
    sum_term = 0.0
    for ni, Si in zip(n_i, cov_i):
        sum_term += (ni - 1) * np.log(np.linalg.det(Si))
    M = (N - g) * np.log(det_sp) - sum_term
    term1 = (2*p*p + 3*p - 1) / (6.0*(p+1)*(g-1))
    term2 = np.sum(1.0/(n_i - 1)) - 1.0/(N - g)
    U = 1.0 - term1 * term2
    chi2_stat = U * M
    df_box = (g - 1) * p * (p + 1) / 2.0
    p_value = 1 - chi2.cdf(chi2_stat, df_box)

    return M, chi2_stat, df_box, p_value
M_val, chi2_val, df_box, p_box = box_m_test(Y, labels_house)
print(f"Box’s M: M={M_val:.1f}, corrected χ²={chi2_val:.1f}, df={df_box:.0f}, p={p_box:.3f}")

M_wk, chi2_wk, df_wk, p_wk = box_m_test(Y, labels_week)
print(f"Box’s M (weekday): M={M_wk:.1f}, χ²={chi2_wk:.1f}, df={df_wk:.0f}, p={p_wk:.3f}")

# ——————————————————————————————
# Independence check
for h in [0, 12, 18]:
    hour_vals = df_smooth_a[f'hour_{h}'].values
    acf_lag1 = np.corrcoef(hour_vals[:-1], hour_vals[1:])[0,1]
    print(f"Hour {h}: lag-1 autocorrelation = {acf_lag1:.3f}")
from statsmodels.tsa.stattools import acf
for h in [0, 12, 18]:
    print(f"\nAuto-correlation (lag1) per household for hour {h}:")
    for hh in groups:
        hv = df_smooth_a[df_smooth_a['factor_household']==hh][f'hour_{h}'].values
        if len(hv)>10:
            print(f"  HH {hh}: {acf(hv, nlags=1)[1]:.3f}")


# MANOVA after normalization
from scipy.stats import chi2, norm
from itertools import combinations
from statsmodels.tsa.stattools import acf

# ——————————————————————————————————————————————————————————————
df_norm = df_smooth_a.copy()
for h in hour_cols:
    df_norm[h] = np.log1p(df_norm[h].values)
Y_norm = df_norm[hour_cols].values
N, p = Y_norm.shape

labels_house = df_norm['factor_household'].values
labels_week  = df_norm['is_weekend'].values

# ——————————————————————————————————————————————————————————————
# 1. Mardia’s
def mardia_test(Y):
    N, p = Y.shape
    mean_vec = np.mean(Y, axis=0)
    S = np.cov(Y, rowvar=False)
    invS = np.linalg.inv(S)
    Z = Y - mean_vec
    g1 = 0.0
    for i in range(N):
        Zi = Z[i]
        for j in range(N):
            Zj = Z[j]
            val = float(np.dot(Zi, invS.dot(Zj)))
            g1 += val**3
    g1 /= (N**2)
    di = np.einsum('ij,jk,ik->i', Z, invS, Z)  # z_i^T invS z_i，依次对每行
    g2 = np.mean(di**2)

    return g1, g2

g1, g2 = mardia_test(Y_norm)
print("=== Mardia’s Multivariate normality test (logarithmized data）=== ")
print(f"  Mardia’s skewness (g1) = {g1:.3f},  kurtosis (g2) = {g2:.3f}")

df_skew = p*(p+1)*(p+2)/6
chi2_skew = n * g1 / 6.0
pval_skew = 1 - chi2.cdf(chi2_skew, df_skew)
print(f"  -> skewness χ² ≈ {chi2_skew:.1f}, df={df_skew:.0f}, p={pval_skew:.4f}")

expected_kurt = p*(p+2)
std_kurt = np.sqrt(8*p*(p+2)/n)
z_kurt = (g2 - expected_kurt) / std_kurt
pval_kurt = 2 * (1 - norm.cdf(abs(z_kurt)))
print(f"  -> kurtosis z ≈ {z_kurt:.3f}, p={pval_kurt:.4f}")

# ——————————————————————————————————————————————————————————————
# 2. Box’s M
def box_m_test(Y, labels):
    unique_labels = np.unique(labels)
    g = len(unique_labels)
    n_i = []
    cov_i = []
    for lab in unique_labels:
        Yi = Y[labels == lab]
        n_i.append(Yi.shape[0])
        cov_i.append(np.cov(Yi, rowvar=False))
    n_i = np.array(n_i)
    N = np.sum(n_i)
    Spooled = np.zeros((p, p))
    for ni, Si in zip(n_i, cov_i):
        Spooled += (ni - 1) * Si
    Spooled = Spooled / (N - g)
    det_sp = np.linalg.det(Spooled)
    sum_term = 0.0
    for ni, Si in zip(n_i, cov_i):
        sum_term += (ni - 1) * np.log(np.linalg.det(Si))
    M_orig = (N - g) * np.log(det_sp) - sum_term
    term1 = (2*p*p + 3*p - 1) / (6.0 * (p + 1) * (g - 1))
    term2 = np.sum(1.0 / (n_i - 1)) - 1.0 / (N - g)
    U = 1.0 - term1 * term2

    chi2_stat = U * M_orig
    df_box = (g - 1) * p * (p + 1) / 2.0
    p_value = 1 - chi2.cdf(chi2_stat, df_box)

    return M_orig, chi2_stat, df_box, p_value

M_val, chi2_val, df_box, p_box = box_m_test(Y_norm, labels_house)
print("\n=== Box’s M Test for homogeneity of covariance (by household group) ===")
print(f"  M = {M_val:.1f}, corrected χ² = {chi2_val:.1f}, df = {df_box:.0f}, p = {p_box:.4f}")
print("  → If p < 0.05，reject null hypothesis")

# 对“is_weekend”分组也可以同理检验
M_wk, chi2_wk, df_wk, p_wk = box_m_test(Y_norm, labels_week)
print("\n=== Box’s M Covariance homogeneity test (grouped by whether it is weekend)  ===")
print(f"  M = {M_wk:.1f}, χ² = {chi2_wk:.1f}, df = {df_wk:.0f}, p = {p_wk:.4f}")

# ——————————————————————————————————————————————————————————————
# 3. Independence
print("\n=== Autocorrelation test (lag-1 autocorrelation within the same household and hour) ===")
for h in [0, 12, 18]:
    print(f"\nHour {h}:")
    for hh in np.unique(labels_house):
        series = df_norm[df_norm['factor_household'] == hh].sort_values('date')[f'hour_{h}'].values
        if len(series) > 10:
            acf1 = acf(series, nlags=1, fft=False)[1]
            print(f"  HH {hh:2d}: lag‐1 ACF = {acf1:.3f}")
        else:
            print(f"  HH {hh:2d}: Insufficient sample size to calculate")

np.random.seed(0)
sample_hh = np.random.choice(df_smooth_a['factor_household'].unique(), 30, replace=False)
df30 = df_smooth_a[df_smooth_a['factor_household'].isin(sample_hh)].copy()

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Task 4
def compute_coef_mat(df, hh_list, filter_weekend=None, interaction=True):
    mats, idxs = [], []
    for hh in hh_list:
        sub = df[df['factor_household'] == hh]
        if filter_weekend is not None:
            sub = sub[sub['is_weekend'] == filter_weekend]
        if len(sub) < 5:
            mats.append(np.zeros(24 * (4 if interaction else 2)))
        else:
            if interaction:
                X = np.column_stack([
                    np.ones(len(sub)),
                    sub['season'].values,
                    sub['is_weekend'].values,
                    (sub['season'] * sub['is_weekend']).values
                    # sub['season'].values
                ])
            else:
                X = np.column_stack([
                    np.ones(len(sub)),
                    sub['season'].values
                ])
            Y = sub[hour_cols].values
            betas = np.linalg.lstsq(X, Y, rcond=None)[0]
            mats.append(betas.T.flatten())
        idxs.append(hh)
    return pd.DataFrame(mats, index=idxs)

# Overall
sample_hh = df30['factor_household'].unique()
sample_hh = [hh for hh in sample_hh if hh not in [19, 21]] # without outlies

coef_full = compute_coef_mat(df30, sample_hh, interaction=True)
scaler_full = StandardScaler().fit(coef_full)
X_full = scaler_full.transform(coef_full)
pca_full = PCA(n_components=0.9, random_state=0).fit(X_full)
X_full_pca = pca_full.transform(X_full)

sil_full = {k: silhouette_score(X_full_pca, KMeans(n_clusters=k,  n_init=50,random_state=0).fit_predict(X_full_pca))
            for k in range(2, 7)}
best_k_full = max(sil_full, key=sil_full.get)
kmeans_full = KMeans(n_clusters=best_k_full, n_init=50,random_state=0).fit(X_full_pca)
labels_full = KMeans(n_clusters=best_k_full, n_init=50,random_state=0).fit_predict(X_full_pca)
summary_full = pd.Series(labels_full, index=coef_full.index, name='cluster') \
    .value_counts().sort_index().rename_axis('cluster').reset_index(name='n_households')

mapping_overall= pd.Series(labels_full, index=coef_full.index)
cluster_dict_overall = mapping_overall.groupby(mapping_overall).apply(lambda grp: grp.index.tolist()).to_dict()

# 2) Separate weekday / weekend clustering with intercept + season model
coef_wd = compute_coef_mat(df30, sample_hh, filter_weekend=0, interaction=False)
coef_we = compute_coef_mat(df30, sample_hh, filter_weekend=1, interaction=False)

def cluster_summary(coefs):
    Xs_fit=StandardScaler().fit(coefs)
    Xs = Xs_fit.transform(coefs)
    pcs_fit=PCA(n_components=0.9, random_state=0).fit(Xs)
    pcs = pcs_fit.transform(Xs)
    sil = {k: silhouette_score(pcs, KMeans(n_clusters=k, random_state=0).fit_predict(pcs))
           for k in range(2, 7)}
    best_k = max(sil, key=sil.get)
    labs_fit=KMeans(n_clusters=best_k, random_state=0).fit(pcs)
    labs = KMeans(n_clusters=best_k, random_state=0).fit_predict(pcs)
    summary = pd.Series(labs, index=coefs.index, name='cluster') \
        .value_counts().sort_index().rename_axis('cluster').reset_index(name='n_households')
    return best_k,Xs_fit,pcs_fit,labs_fit, labs, summary

k_wd, Xs_fit_wd,pcs_fit_wd,labs_fit_wd,labels_wd, summary_wd = cluster_summary(coef_wd)
k_we, Xs_fit_we,pcs_fit_we,labs_fit_we,labels_we, summary_we = cluster_summary(coef_we)

# Compare cluster membership
ct = pd.crosstab(
    pd.Series(labels_wd, index=coef_wd.index, name='weekday_cluster'),
    pd.Series(labels_we, index=coef_we.index, name='weekend_cluster')
)
mapping_wd= pd.Series(labels_wd, index=coef_wd.index)
cluster_dict_wd = mapping_wd.groupby(mapping_wd).apply(lambda grp: grp.index.tolist()).to_dict()

mapping_we= pd.Series(labels_we, index=coef_we.index)
cluster_dict_we = mapping_wd.groupby(mapping_we).apply(lambda grp: grp.index.tolist()).to_dict()

print(f"Clusters chosen: overall={best_k_full}, weekday={k_wd}, weekend={k_we}")


#Task 5
all_hh = df_smooth_a['factor_household'].unique()
all_hh =[hh for hh in all_hh if hh not in [19, 21]] # without outlies
df30_nooutliers=df30[df30['factor_household'].isin(all_hh)]

holdouts = [hh for hh in all_hh if hh not in sample_hh]
coef_hold = compute_coef_mat(df_smooth_a, holdouts,interaction=True)  # same interaction model
Xh = scaler_full.transform(coef_hold)
Xh_pca = pca_full.transform(Xh)
labels_hold = kmeans_full.predict(Xh_pca)

mapping_hold = pd.DataFrame({
    'household': holdouts,
    'assigned_cluster': labels_hold
}).sort_values('household').reset_index(drop=True)

print(mapping_hold)

coef_hold_wd = compute_coef_mat(df_smooth_a, holdouts,filter_weekend=0,interaction=False)  # same interaction model
Xh_wd = Xs_fit_wd.transform(coef_hold_wd)
Xh_pca_wd = pcs_fit_wd.transform(Xh_wd)
labels_hold_wd = labs_fit_wd.predict(Xh_pca_wd)

mapping_hold_wd = pd.DataFrame({
    'household': holdouts,
    'assigned_cluster': labels_hold_wd
}).sort_values('household').reset_index(drop=True)

print(mapping_hold_wd)

coef_hold_we = compute_coef_mat(df_smooth_a, holdouts,filter_weekend=1,interaction=False)  # same interaction model
Xh_we = Xs_fit_wd.transform(coef_hold_we)
Xh_pca_we = pcs_fit_wd.transform(Xh_we)
labels_hold_we = labs_fit_wd.predict(Xh_pca_we)

mapping_hold_we = pd.DataFrame({
    'household': holdouts,
    'assigned_cluster': labels_hold_we
}).sort_values('household').reset_index(drop=True)

print(mapping_hold_we)