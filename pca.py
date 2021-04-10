from sklearn.decomposition import PCA
import pandas as pd

NUM_COMPONENTS = 10

pca_names = ['pca{}'.format(i) for i in range(1, NUM_COMPONENTS + 1)]
spam_base_test_normalize = pd.read_csv('./test_data/spambase-test-normalize.csv')
spam_base_train_normalize = pd.read_csv('./train_data/spambase-train-normalize.csv')
feature_names = spam_base_test_normalize.columns.values[1:]
spam_base_test_normalize = spam_base_test_normalize[feature_names]
spam_base_train_normalize = spam_base_train_normalize[feature_names]


pca = PCA(n_components=NUM_COMPONENTS)

pca_test = pca.fit_transform(spam_base_test_normalize)
pca_train = pca.fit_transform(spam_base_train_normalize)

pcaDF_test = pd.DataFrame(data=pca_test, columns=pca_names)
pcaDF_test = pd.concat([pcaDF_test, spam_base_test_normalize[['class']]], axis=1)
pcaDF_train = pd.DataFrame(data=pca_train, columns=pca_names)
pcaDF_train = pd.concat([pcaDF_train, spam_base_train_normalize[['class']]], axis=1)

pcaDF_test.to_csv('./test_data/pca_spambase_test_pca.csv')
pcaDF_train.to_csv('./train_data/pca_spambase_train_pca.csv')
