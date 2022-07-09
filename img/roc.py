import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest



plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = [10.0, 10.0]
plt.rcParams['figure.dpi'] = 100
sns.set(style="whitegrid", palette="pastel", color_codes=True)

X_train=pd.read_csv("X_train.csv")
X_train.columns = ['X_' + str(i) for i in range(128)]
y_train=pd.read_csv("y_train.csv")
y_train.columns = ['Y']
train_data = pd.concat([X_train, y_train], axis = 1)

N_X_train = (X_train-X_train.min())/(X_train.max()-X_train.min())


XY_train_data = pd.concat([X_train, y_train], axis = 1)

N_XY_train_data = pd.concat([N_X_train, y_train], axis = 1)
X_train.head()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest


def del_high_corr(X_train, corr_threshold):
    del_col = []
    X_train_array = np.array(X_train).T
    ignore = []
    co = np.corrcoef(X_train_array)
    co_index = [[] for i in range(128)]
    _cox, _coy = np.where(abs(co) > corr_threshold)
    for _i in range(len(_cox)):
        if _cox[_i] != _coy[_i]:
            co_index[_cox[_i]].append(_coy[_i])
            ignore.append(_cox[_i])
            if _coy[_i] not in ignore:
                try:
                    del_col.append(_coy[_i])
                except:
                    pass
    return list(set(del_col))


def del_cor_col(df, col):
    df = df.drop(df.columns[col], axis=1)
    df.columns = ['X_' + str(i) for i in range(df.shape[1])]
    return df


def del_outliers(x_df, y_df, contamination):
    clf = IsolationForest(max_samples=1000, random_state=42, n_jobs=-1, contamination=contamination)
    clf.fit(x_df)
    output_table = pd.DataFrame(clf.predict(x_df))
    OL_X_train = x_df.loc[(output_table[0] == 1)]
    OL_X_train.index = [i for i in range(len(OL_X_train))]
    OL_y_train = y_df.loc[(output_table[0] == 1)]
    OL_y_train.index = [i for i in range(len(OL_X_train))]
    OL_X_train = (OL_X_train - OL_X_train.min()) / (OL_X_train.max() - OL_X_train.min())
    OL_XY_train_data = pd.concat([OL_X_train, OL_y_train], axis=1)
    return OL_X_train, OL_y_train


def get_cleaned_data(data_root, corr_threshold=0.99, outlier_contamination='auto'):
    X_train = pd.read_csv(data_root + "X_train.csv")
    X_train.columns = ['X_' + str(i) for i in range(128)]
    y_train = pd.read_csv(data_root + "y_train.csv")
    y_train.columns = ['Y']
    X_val = pd.read_csv(data_root + "X_val.csv")
    y_val = pd.read_csv(data_root + "y_val.csv")
    y_val.columns = ['Y']
    X_test = pd.read_csv(data_root + "X_test.csv")
    N_X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())
    X_train, y_train = del_outliers(N_X_train, y_train, outlier_contamination)
    del_col = del_high_corr(X_train, corr_threshold)
    X_train = del_cor_col(X_train, del_col)
    X_val = del_cor_col(X_val, del_col)
    X_val = (X_val - X_val.min()) / (X_val.max() - X_val.min())
    X_test = del_cor_col(X_test, del_col)
    X_test = (X_test - X_test.min()) / (X_test.max() - X_test.min())
    return X_train, y_train, X_val, y_val, X_test


data_root = ''

X_train, y_train, X_val, y_val, X_test = get_cleaned_data(data_root,
                                                          corr_threshold=0.99,
                                                          outlier_contamination='auto')