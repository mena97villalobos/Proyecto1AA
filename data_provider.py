import pandas as pd


def load_wine_quality_data():
    # Wine Quality dataset, raw data without preprocessing

    wine_quality_train = pd.read_csv('./train_data/winequality-train.csv')
    feature_names = wine_quality_train.columns.values[1:-1]
    features_train_wq = wine_quality_train[feature_names]
    labels_train_wq = [int(i > 5) for i in wine_quality_train['quality'].values]

    wine_quality_test = pd.read_csv('./test_data/winequality-test.csv')
    features_test_wq = wine_quality_test[feature_names]
    labels_test_wq = [int(i > 5) for i in wine_quality_test['quality'].values]
    return features_train_wq, labels_train_wq, features_test_wq, labels_test_wq


def load_wine_quality_data_standard():
    # Wine Quality dataset, standardize data

    wine_quality_train_standardize = pd.read_csv('./train_data/winequality-train-standardize.csv')
    feature_names = wine_quality_train_standardize.columns.values[1:-1]
    features_train_wq_standard = wine_quality_train_standardize[feature_names]
    labels_train_wq_standard = [int(i > 5) for i in wine_quality_train_standardize['quality'].values]

    wine_quality_test_standardize = pd.read_csv('./test_data/winequality-test-standardize.csv')
    features_test_wq_standard = wine_quality_test_standardize[feature_names]
    labels_test_wq_standard = [int(i > 5) for i in wine_quality_test_standardize['quality'].values]
    return features_train_wq_standard, labels_train_wq_standard, features_test_wq_standard, labels_test_wq_standard


def load_wine_quality_data_normal():
    # Wine Quality dataset, normalize data

    wine_quality_train_normalize = pd.read_csv('./train_data/winequality-train-normalize.csv')
    feature_names = wine_quality_train_normalize.columns.values[1:-1]
    features_train_wq_normal = wine_quality_train_normalize[feature_names]
    labels_train_wq_normal = [int(i > 5) for i in wine_quality_train_normalize['quality'].values]

    wine_quality_test_normalize = pd.read_csv('./test_data/winequality-test-normalize.csv')
    features_test_wq_normal = wine_quality_test_normalize[feature_names]
    labels_test_wq_normal = [int(i > 5) for i in wine_quality_test_normalize['quality'].values]
    return features_train_wq_normal, labels_train_wq_normal, features_test_wq_normal, labels_test_wq_normal


def load_spam_base_data():
    # Spam base dataset, raw data without preprocessing

    spam_base_train = pd.read_csv('./train_data/spambase-train.csv')
    feature_names = spam_base_train.columns.values[1:-1]
    features_train_sb = spam_base_train[feature_names]
    labels_train_sb = spam_base_train['class'].values

    spam_base_test = pd.read_csv('./test_data/spambase-test.csv')
    features_test_sb = spam_base_test[feature_names]
    labels_test_sb = spam_base_test['class'].values
    return features_train_sb, labels_train_sb, features_test_sb, labels_test_sb


def load_spam_base_data_standard():
    # Spam base dataset, standardize data

    spam_base_train_standardize = pd.read_csv('./train_data/spambase-train-standardize.csv')
    feature_names = spam_base_train_standardize.columns.values[1:-1]
    features_train_sb_standard = spam_base_train_standardize[feature_names]
    labels_train_sb_standard = spam_base_train_standardize['class'].values

    spam_base_test_standardize = pd.read_csv('./test_data/spambase-test-standardize.csv')
    features_test_sb_standard = spam_base_test_standardize[feature_names]
    labels_test_sb_standard = spam_base_test_standardize['class'].values
    return features_train_sb_standard, labels_train_sb_standard, features_test_sb_standard, labels_test_sb_standard


def load_spam_base_data_normal():
    # Spam base dataset, normalize data

    spam_base_train_normalize = pd.read_csv('./train_data/spambase-train-normalize.csv')
    feature_names = spam_base_train_normalize.columns.values[1:-1]
    features_train_sb_normal = spam_base_train_normalize[feature_names]
    labels_train_sb_normal = spam_base_train_normalize['class'].values

    spam_base_test_normalize = pd.read_csv('./test_data/spambase-test-normalize.csv')
    features_test_sb_normal = spam_base_test_normalize[feature_names]
    labels_test_sb_normal = spam_base_test_normalize['class'].values
    return features_train_sb_normal, labels_train_sb_normal, features_test_sb_normal, labels_test_sb_normal


def load_spam_base_pca():
    pca_spambase_train = pd.read_csv('./train_data/pca_spambase_train_pca.csv')
    pca_spambase_test = pd.read_csv('./test_data/pca_spambase_test_pca.csv')
    feature_names = pca_spambase_train.columns.values[1:-1]

    features_train = pca_spambase_train[feature_names]
    labels_train = pca_spambase_train['class'].values

    features_test = pca_spambase_test[feature_names]
    labels_test = pca_spambase_test['class'].values
    return features_train, labels_train, features_test, labels_test


def data():
    features_train_wq, labels_train_wq, features_test_wq, labels_test_wq = load_wine_quality_data()
    features_train_wq_standard, labels_train_wq_standard, features_test_wq_standard, labels_test_wq_standard = \
        load_wine_quality_data_standard()
    features_train_wq_normal, labels_train_wq_normal, features_test_wq_normal, labels_test_wq_normal = \
        load_wine_quality_data_normal()
    features_train_sb, labels_train_sb, features_test_sb, labels_test_sb = load_spam_base_data()
    features_train_sb_standard, labels_train_sb_standard, features_test_sb_standard, labels_test_sb_standard = \
        load_spam_base_data_standard()
    features_train_sb_normal, labels_train_sb_normal, features_test_sb_normal, labels_test_sb_normal = \
        load_spam_base_data_normal()
    features_pca_train, labels_pca_train, features_pca_test, labels_pca_test = load_spam_base_pca()

    return [
        {
            'train': (features_train_wq, labels_train_wq),
            'test': (features_test_wq, labels_test_wq),
            'label': 'Wine Quality'
        },
        {
            'train': (features_train_sb, labels_train_sb),
            'test': (features_test_sb, labels_test_sb),
            'label': 'Spam Base'
        },
        {
            'train': (features_train_wq_standard, labels_train_wq_standard),
            'test': (features_test_wq_standard, labels_test_wq_standard),
            'label': 'Wine Quality Standard'
        },
        {
            'train': (features_train_sb_standard, labels_train_sb_standard),
            'test': (features_test_sb_standard, labels_test_sb_standard),
            'label': 'Spam Base Standard'
        },
        {
            'train': (features_train_wq_normal, labels_train_wq_normal),
            'test': (features_test_wq_normal, labels_test_wq_normal),
            'label': 'Wine Quality Normal'
        },
        {
            'train': (features_train_sb_normal, labels_train_sb_normal),
            'test': (features_test_sb_normal, labels_test_sb_normal),
            'label': 'Spam Base Normal'
        },
        {
            'train': (features_pca_train, labels_pca_train),
            'test': (features_pca_test, labels_pca_test),
            'label': 'Spam Base PCA (20 components)'
        }
    ]
