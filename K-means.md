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
