# sklearn: ML Library
# %% Classification
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

# ----- (1) Veri Seti İnceleme
cancer = load_breast_cancer() # veri setini içeri aktardık

df = pd.DataFrame(data = cancer.data, columns= cancer.feature_names)
# veri setini dataframe'e çevirdik

df["target"] = cancer.target
# veri setinde target sütunu yoktu, ekledik.


# ----- (2) Makine Öğrenmesi Modelinin Seçilmesi - KNN Sınıflandırıcı
# ----- (3) Modelin Train Edilmesi
X = cancer.data # features - özellikler
y = cancer.target # target - hedef

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
# X_train, y_train 'ler train dataları
# X_test, y_test 'ler test dataları

# ölçeklendirme
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors = 9) # K değerini yaz.
# Model oluşturma

knn.fit(X_train, y_train) # Modeli fit etme. fit fonksiyonu verimizi (samples + target) kullanarak knn algoritmasını eğitir.

# ----- (4) Sonuçların Değerlendirilmesi
y_pred = knn.predict(X_test) # verimizi (samples, features) verdik ve verinin sonucunu (target) tahmin etti

accuracy = accuracy_score(y_test, y_pred) # test verileri ile tahmin ettiği değerler arasındaki doğruluk oranını hesaplar
print("Doğruluk: ", accuracy) # %94,72759226713533 başarılı olmuş bu model

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion_matrix: ", conf_matrix) # matris şekilnde değerlerin ne kadarını doğru ne kadarını yanlış tahmin ettiğini gösterir.

# ----- (5) Hiperparametre Ayarlaması
# yapma amacımız modelin doğruluğunu artırmak.
"""
    KNN: Hyperparameter = K
        K: 1, 2, 3, ... N
        Accuracy: %A, %B, %C .....
"""

accuracy_values = []
k_values = []
for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_values.append(accuracy)
    k_values.append(k)
    
plt.figure()
plt.plot(k_values, accuracy_values, marker = "o", linestyle = "-")
plt.title("k değerine göre doğruluk")
plt.xlabel("K değeri")
plt.ylabel("Doğruluk")
plt.xticks(k_values)
plt.grid(True)


# %% regression
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

X = np.sort(5 * np.random.rand(40, 1), axis = 0) # uniform, features
y = np.sin(X).ravel()

# plt.scatter(X,y)
T = np.linspace(0, 5, 500)[:, np.newaxis]

# add noise
y[::5] += 1 * (0.5 - np.random.rand(8))
# plt.scatter(X,y)

for i , weight in enumerate(["uniform", "distance"]):
    
    knn = KNeighborsRegressor(n_neighbors=5, weights=weight)
    y_pred = knn.fit(X, y).predict(T)

    plt.subplot(2, 1, i+1)
    plt.scatter(X, y, color = "green", label = "data")
    plt.plot(T, y_pred, color = "blue", label = "prediction")
    plt.axis("tight")
    plt.legend()
    plt.title("KNN Regressor weights = {}".format(weight))
    
plt.tight_layout()
plt.show()










































