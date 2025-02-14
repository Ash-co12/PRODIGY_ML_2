import pandas as pnd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler  
from sklearn.cluster import KMeans

#reading using pandas
try:
    td = pnd.read_csv("Mall_Customers.csv")
except FileNotFoundError:
    print("No such file exist please put the Mall_Customers.csv File in the folder with main.py")

allspend = td[["Age","Annual Income (k$)","Spending Score (1-100)"]]

#Scaling
ss = StandardScaler()
sd = ss.fit_transform(allspend)

#clustering and predicting
kmc = KMeans(n_clusters=5)
tl = kmc.fit_predict(sd)

#adding a clusters column to training dataset and initializing 3d plotting
td['clusters'] = tl
fig = plt.figure()
az = plt.axes(projection='3d')

for i in range(5):
    cluster_data = td[td['clusters'] == i]
    az.scatter(cluster_data['Age'], cluster_data['Annual Income (k$)'], cluster_data['Spending Score (1-100)'], label=f'Cluster {i}')

az.set_xlabel('Age')
az.set_ylabel('Annual Income (k$)')
az.set_zlabel('Spending Score (1-100)')
az.set_title('K-Means Clustering (Age, Income, Spending Score)')
az.legend()
plt.show()
