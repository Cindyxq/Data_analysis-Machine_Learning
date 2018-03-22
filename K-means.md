# K-Means Algorithm

Machine Learning:
* Supervised learning
* Unsupervised learning
* Semi-supervised learning

## 监督学习（Supervised learning)
从给定的训练数据集中学习出一个函数，当新的数据到来时，可以根据这个函数预测结果。监督学习的训练集要求包括输入输出，也可以说是特征和目标。
常见的监督学习算法：回归分析和统计分类。 KNN(K Nearest Neighbor)，SVM(Support Vector Machine)

## 无监督学习（Unsupervised learning)
输入数据没有被标记，也没有确定的结果。样本数据类别未知，需要根据样本间的相似性对样本集进行分类(聚类，clustering) 试图使类内差距最小化，类间差距最大化。
使用聚类结果，可以提取数据集中隐藏信息，对未来数据进行分来和预测。应用于数据挖掘，模式识别，图像处理等。
PCA（Principle Component Analysis) 和很多deep learning算法都属于无监督学习

## K-Means Algorithm
Cluster is a type of **Unsupervised Learning**. This is very often used when you don't have labeled data. 
**K-Means Clustering** is one of the popular clustering algorithm. The goal of this algorithm is to find groups(clusters)
in the given data. 

K-Means is a very simple algorithm which clusters the data into K number of clusters. The following image is an example of K-Means Clustering.

![Clustering](cluster.jpg)   

### Algorithm
Assuming we have inputs $x_1, $x_2, $x_3, ..., $x_n and value of **K**
* **Step 1** - Pick K random points as cluster centers called centroids.
* **Step 2** - Assign each $x_i to nearest cluster by calculating its distance to each centroid
* **Step 3** - Find new cluster center by taking the average of the assigned points
* **Step 4** - Repeat Step 2 and 3 until none of the cluster assignments change

[Python implementation](https://github.com/mubaris/friendly-fortnight)

How to choose the K value? [Elbow Method](https://pythonprogramminglanguage.com/kmeans-elbow-method/)

### 使用场景
不清楚用户有几类时，尝试性的将用户进行分类，并根据每类用户的不同特征，决定下一步动作。
举例，对于一个超市/电商网站/综合零售商，可以根据用户的购买行为，将其分为“年轻白领”、“一家三口”、“家有一老”、”初得子女“等等类型，然后通过邮件、短信、推送通知等，向其发起不同的优惠活动。

[Andrew Ng K-Means](https://www.coursera.org/learn/machine-learning/lecture/93VPG/k-means-algorithm
)

## Python data analysis library  (Anaconda)
* [numpy](https://docs.scipy.org/doc/numpy-1.13.0/user/quickstart.html)
  以矩阵为基础的数学计算模块，纯数学存储和处理大型矩阵。
* [pandas](https://pandas.pydata.org/)
  数据分析。基于Numpy的一种工具，为了解决数据分析任务而创建的。纳入了大量的库和一些标准的数据模型，提供了高效地操作大型数据集所需的工具。
  Series, DataFrame
* [matplotlib](https://matplotlib.org/)
  [seaborn](https://seaborn.pydata.org/)
  绘图
* [scipy](https://www.scipy.org/)
  数值计算库，在Numpy库的基础上增加了众多的数学，科学以及工程计算中常用的库函数。它包括统计,优化,整合,线性代数模块,傅里叶变换,信号和图像处理,常微分方程求解器等等。
* [sklearn](http://scikit-learn.org/stable/)
  Machine Learning in Python<br>
  eg: from sklearn.cluster import **KMeans**

