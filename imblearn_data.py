from collections import Counter

from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

from imblearn.datasets import make_imbalance
from imblearn.under_sampling import NearMiss
from imblearn.pipeline import make_pipeline
from imblearn.metrics import classification_report_imbalanced

RANDOM_STATE = 42

iris = load_iris()
X, y = make_imbalance(iris.data, iris.target, ratio = {0:25, 1:50, 2:50}, random_state = 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = RANDOM_STATE)

print ('Training target statistics: {}'.format(Counter(y_train)))
print ('Testing target statistics: {}'.format(Counter(y_test)))

pipeline = make_pipeline(NearMiss(version = 2, random_state = RANDOM_STATE), LinearSVC(random_state = RANDOM_STATE))
pipeline.fit(X_train, y_train)

print (classification_report_imbalanced(y_test, pipeline.predict(X_test)))