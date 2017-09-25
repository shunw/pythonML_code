from sklearn.datasets import make_classification
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

from imblearn.datasets import make_imbalance
from imblearn.under_sampling import RandomUnderSampler

print (__doc__)

def plot_pie(y):
    target_stats = Counter(y)
    labels = list(target_stats.keys())
    
    sizes = list(target_stats.values())
    
    explode = tuple([.1] * len(target_stats))

    fig, ax = plt.subplots()
    ax.pie(sizes, explode = explode, labels = labels, shadow = True, autopct = '%1.1f%%')
    ax.axis('equal')

def ratio_multiplier(y):
    multiplier = {0:.5, 1:.7, 2:.95}
    target_stats = Counter(y)
    for key, value in target_stats.items():
        target_stats[key] = int(value * multiplier[key])
    return target_stats

if __name__ == '__main__':
    iris = load_iris()
    print('Information of the original iris data set: \n {}'.format(Counter(iris.target)))
    plot_pie(iris.target)

    ratio = {0:10, 1:20, 2:30}
    X, y = make_imbalance(iris.data, iris.target, ratio = ratio)
    print('Information of the iris data set after making it imbalanced using a dict: \n ratio = {} \n y: {}'.format(ratio, Counter(y)))
    plot_pie(y)

    X, y = make_imbalance(iris.data, iris.target, ratio = ratio_multiplier(iris.target))
    print('Information of the iris data set after making it'
      ' imbalanced using a callable: \n ratio={} \n y: {}'.format(
          ratio_multiplier, Counter(y)))
    plot_pie(y)

    '''use ratio in resampling algorithm'''
    ratio = 'minority'
    print (Counter(y))
    X_res, y_res = RandomUnderSampler(ratio = ratio, random_state = 0).fit_sample(X, y)
    print (ratio)
    print (Counter(y_res))

    # plt.show()



