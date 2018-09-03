from sklearn.feature_extraction import DictVectorizer
import  csv
from sklearn import preprocessing
from sklearn import tree
import  random
from sklearn.externals.six import StringIO

f=open("ihc_sample.csv","r")
r=csv.reader(f)
header=next(f).split(",")

featurelist=[]
labellist=[]
for row in r:

    labellist.append(row[len(row)-1])
    rowDict={}
    for i in range(1,len(row)-1):
        rowDict[header[i]]=row[i]
    featurelist.append(rowDict)

vec=DictVectorizer()
data=vec.fit_transform(featurelist).toarray()

lb=preprocessing.LabelBinarizer()
label=lb.fit_transform(labellist)
print(labellist)


clf=tree.DecisionTreeClassifier(criterion="entropy")
decesiontree=clf.fit(data,label)
with open("DT.dot","w") as f:
    f=tree.export_graphviz(clf,feature_names=vec.get_feature_names(),out_file=f)


#prediction
#
testindex=random.sample([i for i in range(1,len(labellist)-1)],50)
testdata=[]
testlabel=[]
for i in testindex:
    testdata.append(data[i])
    testlabel.append(label[i])
print(testdata)
print(testlabel)
predict=clf.predict(testdata)
print("predict:",predict)
correctnum=0
for i in range(len(testdata)):
    if predict[i]==testlabel[i]:
        correctnum+=1
print("Accurray:",correctnum/len(testdata))

