# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries,read the data set.find the number of null data.

2.Find the number of null data.

3.Import sklearn library.

4.Find y predict and plot the graph.

## Program:
```
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Jeyabalan
RegisterNumber:  212222240040

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("Mall_Customers (1).csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.cluster import KMeans
wcss=[]

for i in range (1,11):
    kmeans=KMeans(n_clusters = i,init="k-means++")
    kmeans.fit(data.iloc[:,3:])
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.xlabel("No. of clusters")
plt.ylabel("wcss")
plt.title("Elbow matter")

km=KMeans(n_clusters=5)
km.fit(data.iloc[:,3:])

y_pred=km.predict(data.iloc[:,3:])
y_pred

data["cluster"]=y_pred
df0=data[data["cluster"]==0]
df1=data[data["cluster"]==1]
df2=data[data["cluster"]==2]
df3=data[data["cluster"]==3]
df4=data[data["cluster"]==4]
plt.scatter(df0["Annual Income (k$)"],df0["Spending Score (1-100)"],c="red",label="cluster0")
plt.scatter(df1["Annual Income (k$)"],df1["Spending Score (1-100)"],c="black",label="cluster1")
plt.scatter(df2["Annual Income (k$)"],df2["Spending Score (1-100)"],c="blue",label="cluster2")
plt.scatter(df3["Annual Income (k$)"],df3["Spending Score (1-100)"],c="green",label="cluster3")
plt.scatter(df4["Annual Income (k$)"],df4["Spending Score (1-100)"],c="magenta",label="cluster4")
plt.legend()
plt.title("Customer Segmets")
```

## Output:

## data.head():
![image](https://github.com/jeyaqbalan7/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119393851/be8a8e96-aefa-41e7-ab7f-46fdfa14164f)

## data.info():
![image](https://github.com/jeyaqbalan7/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119393851/fd1bdf1a-0367-4f17-9d24-480e01b28e1e)

## data.isnull().sum():
![image](https://github.com/jeyaqbalan7/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119393851/04deba86-8229-4651-8fbb-3600c645fc10)

## Elbow method:
![image](https://github.com/jeyaqbalan7/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119393851/bbea7ddd-76c0-4534-be47-bf2d8974a629)

## K-Means:
![image](https://github.com/jeyaqbalan7/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119393851/f9b0cc76-0930-45dc-9c06-0e66f22fa2e8)

## Array value of Y:
![image](https://github.com/jeyaqbalan7/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119393851/34d0e2a3-df2d-4f17-b337-f9794202f721)

## Customer Segmentation:
![image](https://github.com/jeyaqbalan7/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119393851/5a13241a-8446-4bf1-ad34-4d4287ee3afd)

## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
