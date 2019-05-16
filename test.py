import pandas
import statsmodels.formula.api as smf

data = pandas.read_csv('E:\\brain_size.csv', sep=";", na_values=".")
models = smf.ols("VIQ ~ Gender + 1", data).fit()
print(models.summary())
