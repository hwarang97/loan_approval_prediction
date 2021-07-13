import pandas as pd

# 새로운 데이터를 구해왔음 ( 출저 : https://www.kaggle.com/zhijinzhai/loandata )
train_df = pd.read_csv('input/train.csv')
train_df.info()