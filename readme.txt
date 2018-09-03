实现过程
1.数据转换
sklearn的输入：feature和lable都必须是数值型值,因此要将其他类型转成数值型。
feature转换：

vec=DictVectorizer()
data=vec.fit_transform(featurelist).toarray()
featurelist是一个由字典组成的列表，每个字典都是每一行各个列和值组成和字典{列：值,列：值.........}
label转换:
lb=preprocessing.LabelBinarizer()
label=lb.fit_transform(labellist)
labellist是每行的label组成的列表
2.建模
clf=tree.DecisionTreeClassifier(criterion="entropy")
decesiontree=clf.fit(data,label)
#保存模型
with open("DT.dot","w") as f:
    f=tree.export_graphviz(clf,feature_names=vec.get_feature_names(),out_file=f)
3.预测
#随机从训练集中选取50条，作为测试样本
predict=clf.predict(testdata)