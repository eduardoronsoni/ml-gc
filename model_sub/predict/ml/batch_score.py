# %%
import pandas as pd
import sqlalchemy

import argparse

#

# %%

date = 'max'

print('Importando modelo ...')
model = pd.read_pickle('../../../models/modelo_subscription.pkl')
print('ok')

print('Importando query ...')
with open('C:/Users/eduar/OneDrive/Área de Trabalho/ranked-ml-main/model_sub/predict/etl/query.sql', 'r') as open_file:
    query = open_file.read()
print('ok')
# %%
query

# %%
print('Obtendo data para escoragem ...')
con = sqlalchemy.create_engine(
    'sqlite:///C:/Users/eduar/OneDrive/Área de Trabalho/ranked-ml-main/data/gc.db')

if date == 'max':
    date = pd.read_sql('SELECT MAX(dtRef) as date FROM tb_book_players', con)[
        'date'][0]
else:
    date = date
print('ok')

print('Importando dados ...')
query = query.format(date=date)
df = pd.read_sql(query, con)
print('ok')

# %%
print('Realizando o score dos dados ...')
df_score = df[['dtRef', 'idPlayer']]
df_score['score'] = model['model'].predict_proba(df[model['features']])
df_score.head()
print('ok')
