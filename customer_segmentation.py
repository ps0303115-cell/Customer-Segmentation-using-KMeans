import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import joblib

df = pd.read_csv("data/Mall_Customers.csv")

print(df.head())

X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1,11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
y_pred = kmeans.fit_predict(X)

joblib.dump(kmeans, "model.pkl")

plt.figure(figsize=(8,6))

plt.scatter(X.iloc[:,0], X.iloc[:,1], c=y_pred, cmap='viridis', s=50)

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=300, c='red', marker='X', label='Centroids')

plt.title("Customer Segmentation using K-Means")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()

plt.show(block=False)
plt.pause(3)
plt.close()

print("Model saved successfully!")
