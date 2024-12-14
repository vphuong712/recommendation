import pandas as pd
import numpy as np
# from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfTransformer
from read_data import n_users, rate_train, rate_test, rates, items, n_items

X0 = items.values
X_train_counts = X0[:, 2:]
# print(X_train_counts )
transformer = TfidfTransformer(smooth_idf=True, norm ='l2')
tfidf = transformer.fit_transform(X_train_counts.tolist()).toarray()
# print(tfidf)

def get_items_rated_by_user(rate_matrix, user_id):
    # Lọc các hàng có user_id khớp
    user_ratings = rate_matrix[rate_matrix[:, 0] == user_id]
    
    # Lấy product_id (giảm 1 để đưa về dạng chỉ số bắt đầu từ 0)
    item_ids = user_ratings[:, 1].astype(int) - 1
    
    # Lấy rating
    scores = user_ratings[:, 2]
    
    return (item_ids, scores)

print(get_items_rated_by_user(rate_train, 900))


class Contentbased:
    def __init__(self, Y, X_train, n_users, n_items, lamda = 1):
        self.Y = Y
        self.lamda = lamda
        self.X_train = X_train
        self.n_users = n_users
        self.n_items = n_items

    def fit(self):
        transformer = TfidfTransformer(smooth_idf=True, norm='l2')
        tfidf = transformer.fit_transform(self.X_train.tolist()).toarray()
        d = tfidf.shape[1]  # Data dimension
        W = np.zeros((d, self.n_users))
        b = np.zeros((1, self.n_users))
        
        for n in range(self.n_users):
            ids, scores = get_items_rated_by_user(self.Y, n)
            if np.isnan(scores).any() or len(ids) == 0:  # If user has no ratings or NaN scores
                print(f"User {n} has no rated items or NaN scores. Skipping...")
                continue  # Skip this user
            clf = Ridge(alpha=self.lamda, fit_intercept=True)
            Xhat = tfidf[ids, :]
            clf.fit(Xhat, scores)
            W[:, n] = clf.coef_
            b[0, n] = clf.intercept_
        self.Yhat = tfidf.dot(W) + b


    def RMSE(self, Data_test):
        se = cnt = 0
        for n in range(self.n_users):
            ids, scores_truth = get_items_rated_by_user(Data_test, n)
            scores_pred = self.Yhat[ids, n]
            e = scores_truth - scores_pred
            se += (e*e).sum(axis = 0)
            cnt += e.size
        return np.sqrt(se/cnt)

    def recommend(self, user_id, top):
        a = np.zeros((self.n_items,))
        recommended_items = []
        items_rated_by_user, score = get_items_rated_by_user(self.Y, user_id)
        for i in range(self.n_items):
            if i not in items_rated_by_user:
                a[i] = self.Yhat[i, user_id]
        if len(a) < top:
            recommended_items = np.argsort(a)[-len(a):]
        else:
            recommended_items = np.argsort(a)[-top:]
        return recommended_items

def test():
    for lamda in [1, 3, 5, 7, 9]:
        cb = Contentbased(rates, X_train_counts, n_users= 10, n_items = n_items, lamda=lamda)
        cb.fit()
        RMSE = cb.RMSE(Data_test=rate_train)
        print(f"lamda {lamda}: ",RMSE)
        _lamda = min(lamda, RMSE)
    cb = Contentbased(rates, X_train_counts, n_users= n_users, n_items = n_items, lamda=_lamda)
    cb.fit()
    print(cb.RMSE(rates))
    rcm_list =  list(cb.recommend(99, 10))
    rcm_movie = items[items['product_id'].isin(rcm_list)]['product_name']
    print(rcm_movie)
    
test()         