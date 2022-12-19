import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import accuracy_score
import pickle

DATA_DIR = '../data'
MODEL_DIR = '../models'
SEED = 42  # used to fix random_state
output_file = MODEL_DIR + '/XgboostClassifier.bin'


df_train = pd.read_csv(DATA_DIR + '/processed/train.csv', index_col=0)
df_val = pd.read_csv(DATA_DIR + '/processed/val.csv', index_col=0)
df_test = pd.read_csv(DATA_DIR + '/processed/test.csv', index_col=0)

data_test = df_test.copy()
data_test.reset_index(drop=True)

y_test = data_test['music_genre'].values

del data_test['music_genre']

dv = DictVectorizer(sparse=False)
LE = LabelEncoder()


# training the final model

df_full_train = pd.concat([df_train, df_val]).reset_index(drop = True)
y_full_train = df_full_train['music_genre'].values

del df_full_train['music_genre']


df_full_train_dict = df_full_train.to_dict(orient='records')
X_full_train = dv.fit_transform(df_full_train_dict)
Y_full_train = LE.fit_transform(y_full_train)

test_dict = data_test.to_dict(orient='records')
X_test = dv.transform(test_dict)
Y_test = LE.transform(y_test)

print('training the final model')

model = xgb.XGBClassifier(seed=SEED)
model.fit(X_full_train, Y_full_train)

y_pred = model.predict(X_test)
print(f'score = {accuracy_score(Y_test, y_pred)}')


# saved model

output_file = MODEL_DIR + '/XgboostClassifier.bin'

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, LE, model), f_out)

print(f'the model is saved to {output_file}')


