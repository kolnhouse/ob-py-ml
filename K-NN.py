# ライブラリのインポート
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

# データセットの読み込み
df = pd.read_csv("Social_Network_Ads.csv")

# ラベルエンコーディング処理
le=LabelEncoder()
df['Gender']=le.fit(df['Gender']).transform(df['Gender'])

# 正規化
scaler = MinMaxScaler()
df.loc[:,:]  = scaler.fit_transform(df)
df.head()

# 特徴量と目的変数の選定
X = df[["Gender","Age", "EstimatedSalary"]]
y  = df["Purchased"]

# テストデータ分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# KNNのインスタンス定義
knn = KNeighborsClassifier(n_neighbors=6)

# モデルfit
knn.fit(X,y)

#スコア計算
score = format(knn.score(X_test, y_test))
print('正解率:', score)
