ʵ�ֹ���
1.����ת��
sklearn�����룺feature��lable����������ֵ��ֵ,���Ҫ����������ת����ֵ�͡�
featureת����

vec=DictVectorizer()
data=vec.fit_transform(featurelist).toarray()
featurelist��һ�����ֵ���ɵ��б�ÿ���ֵ䶼��ÿһ�и����к�ֵ��ɺ��ֵ�{�У�ֵ,�У�ֵ.........}
labelת��:
lb=preprocessing.LabelBinarizer()
label=lb.fit_transform(labellist)
labellist��ÿ�е�label��ɵ��б�
2.��ģ
clf=tree.DecisionTreeClassifier(criterion="entropy")
decesiontree=clf.fit(data,label)
#����ģ��
with open("DT.dot","w") as f:
    f=tree.export_graphviz(clf,feature_names=vec.get_feature_names(),out_file=f)
3.Ԥ��
#�����ѵ������ѡȡ50������Ϊ��������
predict=clf.predict(testdata)