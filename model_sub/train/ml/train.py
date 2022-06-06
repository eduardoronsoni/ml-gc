# %%
import pandas as pd
import sqlalchemy

from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn import pipeline
from sklearn import metrics
from sklearn import tree
from sklearn.model_selection import GridSearchCV


from feature_engine import imputation
from feature_engine import encoding

import scikitplot as skplt

import matplotlib.pyplot as plt

# %%
# SAMPLE

con = sqlalchemy.create_engine(
    "sqlite:///C:/Users/eduar/OneDrive/Área de Trabalho/ranked-ml-main/data/gc.db")
con.table_names()

df = pd.read_sql_table('tb_abt_sub', con)

df.head()

# separa como backtest com as 2 ultimas datas para não usar no modelo
df_oot = df[df['dtRef'].isin(['2022-01-15', '2021-01-16'])].copy()

df_train = df[~ df['dtRef'].isin(['2022-01-15', '2021-01-16'])].copy()


# definir aleatoriamente base de treino e teste
features = df_train.columns.tolist()[2:-1]
target = 'flagSub'


X_train, X_test, y_train, y_test = train_test_split(df_train[features],
                                                    df_train[target], random_state=42, test_size=0.2)


# EXPLORE
# análise exploratória apenas em cima de x_train - ser o mais fiel possível
# %%

cat_features = X_train.dtypes[X_train.dtypes == 'object'].index.tolist()
num_features = list(set(X_train.columns) - set(cat_features))

print('Missing Numérico')
is_na = X_train[num_features].isna().sum()
print(is_na[is_na > 0])


missing_0 = ['avgKDA']
missing_1 = ['winRateVertigo',
             'winRateAncient',
             'winRateTrain',
             'winRateOverpass',
             'winRateDust2',
             'winRateNuke',
             'winRateInferno',
             'vlIdade',
             'winRateMirage']

# %%

print('Missing Categórico')
is_na = X_train[cat_features].isna().sum()
print(is_na[is_na > 0])


# %%


# MODIFY

# imputação de dados
imput_0 = imputation.ArbitraryNumberImputer(
    arbitrary_number=0, variables=missing_0)

imput_1 = imputation.ArbitraryNumberImputer(
    arbitrary_number=-1, variables=missing_1)


# one hot encoding
onehot = encoding.OneHotEncoder(drop_last=True, variables=cat_features)

# MODEL

rf_clf = ensemble.RandomForestClassifier(
    n_estimators=200, min_samples_leaf=50, n_jobs=-1, random_state=42)

ada_clf = ensemble.AdaBoostClassifier(n_estimators=200,
                                      learning_rate=0.8,
                                      random_state=42)

dt_clf = tree.DecisionTreeClassifier(max_depth=15,
                                     min_samples_leaf=50,
                                     random_state=42)

rl_clf = linear_model.LogisticRegressionCV(cv=4, n_jobs=-1)

params = {'n_estimators': [50, 100, 200, 250],
          'min_samples_leaf': [5, 10, 20, 50, 100]}

grid_search = GridSearchCV(rf_clf,
                           params,
                           n_jobs=1,
                           cv=4,
                           scoring='roc_auc',
                           verbose=3,
                           refit=True)


# Definir um pipeline

pipe_rf = pipeline.Pipeline(steps=[('Imput 0', imput_0),
                                   ('Imput 1', imput_1),
                                   ('One Hot', onehot),
                                   ('Modelo', grid_search)])


# %%

def train_test_report(model, X_train, y_train, X_test, y_test, key_metric, is_prob=True):

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)

    metric_result = key_metric(
        y_test, prob[:, 1]) if is_prob else key_metric(y_test, pred)
    return metric_result


# %%
pipe_rf.fit(X_train, y_train)

# %%

y_test_pred = model_pipe.predict(X_test)
y_test_prob = model_pipe.predict_proba(X_test)

acc_test = metrics.accuracy_score(y_test, y_test_pred)
roc_test = metrics.roc_auc_score(y_test, y_test_prob[:, 1])

print('acc_test:', acc_test)
print('roc_test:', roc_test)

print('Baseline:', round((1 - y_test.mean())*100, 2))
print('Acurácia:', acc_test)
# %%

skplt.metrics.plot_roc(y_test, y_test_prob)
plt.show()

# %%

skplt.metrics.plot_ks_statistic(y_test, y_test_prob)
plt.show()

# %%

skplt.metrics.plot_precision_recall(y_test, y_test_prob)
plt.show()
# %%

skplt.metrics.plot_lift_curve(y_test, y_test_prob)
plt.show()
# %%
features_model = model_pipe[:-1].transform(X_train.head()).columns.tolist()

fs_importance = pd.DataFrame({'importance': model_pipe[-1].feature_importances_,
                              'feature': features_model})

fs_importance.sort_values('importance', ascending=False).head(20)

# %%
fs_importance.head()


# %%
