import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets
import numpy as np

def generate_data(n_samples, flagc):
    
    if flagc == 1:
        random_state = 365
        X,y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
        
    elif flagc == 2:
        random_state = 148
        X,y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X = np.dot(X, transformation)
        
    elif flagc == 3:
        random_state = 148
        X, y = datasets.make_blobs(n_samples=n_samples,
                                    centers=4,
                                    cluster_std=[1.0, 2.5, 0.5, 3.0],
                                    random_state=random_state)

    elif flagc == 4:
        X, y = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
        
    elif flagc == 5:
        X, y = datasets.make_moons(n_samples=n_samples, noise=.05)
    
    else:
        X = []
        
    return X


for i in range (1,6):
    data = generate_data(500, i)

    kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(data)
    kmeans.fit(data)
    centers=kmeans.cluster_centers_
   
    plt.scatter(data[:,0],data[:,1],marker="o",c=kmeans.labels_)
    plt.scatter(centers[:,0],centers[:,1], c='r')
    #print(centers)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Prvi zadatak')
    plt.show()

    inertias = []
    numCentara = range(1, 10)

    for k in numCentara: 
        kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(data)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
plt.plot(numCentara, inertias, marker='o')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Inercija')
plt.show()

