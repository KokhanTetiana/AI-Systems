import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('renfe_small.csv')
df_clean = df.dropna(subset=['price', 'train_type', 'fare']).copy()

df_clean = df_clean.sample(n=200, random_state=42)

df_clean['train_type_code'] = pd.Categorical(df_clean['train_type']).codes
df_clean['fare_code'] = pd.Categorical(df_clean['fare']).codes

train_type_labels = pd.Categorical(df_clean['train_type']).categories
fare_labels = pd.Categorical(df_clean['fare']).categories

print("Дані успішно підготовлені.")
print(f"Категорії типів поїздів: {list(train_type_labels)}")
print(f"Категорії тарифів: {list(fare_labels)}")

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.histplot(np.log1p(df_clean['price']), kde=True)
plt.title('Розподіл лог-цін на квитки')
plt.xlabel('log(Ціна + 1)')
plt.ylabel('Частота')

plt.subplot(1, 2, 2)
sns.boxplot(x='train_type', y=np.log1p(df_clean['price']), data=df_clean)
plt.title('Лог-ціна залежно від типу поїзда')
plt.xlabel('Тип поїзда')
plt.ylabel('log(Ціна + 1)')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

with pm.Model() as renfe_model:
    beta0 = pm.Normal('beta0', mu=0, sigma=10)
    beta_train_type = pm.Normal('beta_train_type', mu=0, sigma=5, shape=len(train_type_labels))
    beta_fare = pm.Normal('beta_fare', mu=0, sigma=5, shape=len(fare_labels))
    sigma = pm.HalfNormal('sigma', sigma=5)

    mu = beta0 + beta_train_type[df_clean['train_type_code']] + beta_fare[df_clean['fare_code']]

    price_obs = pm.Normal('price_obs', mu=mu, sigma=sigma, observed=np.log1p(df_clean['price']))

    idata = pm.sample(500, tune=250, cores=1, random_seed=42, progressbar=True)
    idata.extend(pm.sample_posterior_predictive(idata, random_seed=42))

print("Модель успішно навчена.")

summary = az.summary(idata, var_names=['beta0', 'beta_train_type', 'beta_fare', 'sigma'], hdi_prob=0.94)
print("\nЗведення параметрів моделі:")
print(summary)

az.plot_ppc(idata, num_pp_samples=100)
plt.title('Апостеріорна предиктивна перевірка (лог-ціни)')
plt.show()
