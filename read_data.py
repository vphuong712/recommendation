import pandas as pd
import numpy as np

file_path = r'data\ml-latest-small'
users = pd.read_csv(file_path+'/users.csv', encoding='latin-1')
ratings = pd.read_csv(file_path+'/ratings.csv', encoding='latin-1')
movies = pd.read_csv(file_path+'/movies.csv', encoding='latin-1')


n_users = users.shape[0]
print('Number of users:', n_users)

ratings_base = pd.read_csv(file_path+'/train_data.csv', encoding='latin-1')
ratings_test = pd.read_csv(file_path+'/test_data.csv', encoding='latin-1')

rate_train = ratings_base.values
rate_test = ratings_test.values

print ('Number of traing rates:', rate_train.shape[0])
print ('Number of test rates:', rate_test.shape[0])

items = pd.read_csv(file_path+'/movies.csv', encoding='latin-1')
n_items = items.shape[0]
print('Number of items:', n_items)