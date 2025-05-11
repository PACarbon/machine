from sklearn.datasets import load_iris    #加载鸢尾花（Iris）数据集
from sklearn.model_selection import cross_val_score     #cross_val_score 是用来做交叉验证的函数
from sklearn.neighbors import KNeighborsClassifier      #KNN 模型（K-最近邻）的类
from sklearn.tree import DecisionTreeClassifier         #决策树模型的类
from sklearn.model_selection import KFold          #KFold 是一个类，意思是“K 折交叉验证”中的折叠规则
 
 
# 加载鸢尾花数据集
iris = load_iris()
X = iris.data       #所有输入特征
y = iris.target     #每朵花的标签
 
# 初始化KNN和决策树模型
knn_model = KNeighborsClassifier()     # KNN 分类器
dt_model = DecisionTreeClassifier()     #决策树分类器
 
 
#将数据集划分为5个折叠，对数据进行洗牌，并设置随机种子为0
kf = KFold(n_splits=5, shuffle=True, random_state=0)
 
# 使用5折交叉验证比较两个模型的性能
knn_scores = cross_val_score(knn_model, X, y, cv=kf)
dt_scores = cross_val_score(dt_model, X, y, cv=kf)
 
print("KNN模型准确率：", knn_scores.mean())
print("决策树模型准确率：", dt_scores.mean())
