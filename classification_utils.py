from scipy.stats import chi2_contingency
from pandas import crosstab


def chi_test(df, columns, target):
    results_dict = {"stat":[], "p":[], "dof":[], "expected":[]}
    correlated_features = []
    uncorrelated_features = []
    
    for col in columns:
        cont_table = crosstab(index = df[col], columns = target)
        stat, p, dof, expected = chi2_contingency(cont_table)
        results_dict["stat"].append(stat)
        results_dict["p"].append(p)
        results_dict["dof"].append(dof)
        results_dict["expected"].append(expected)
        
        if p < 0.05:
            # null hypothesis is rejected in favour of the alternative
            correlated_features.append(col)
        else:
            uncorrelated_features.append(col)
        
    return correlated_features, uncorrelated_features, results_dict


def chi_test_new(df, columns, target, confidence_lvl=0.05):
    results_dict = {}
    correlated_features = []
    uncorrelated_features = []
    
    for col in columns:
        cont_table = crosstab(index = df[col], columns = target)
        stat, p, dof, _ = chi2_contingency(cont_table)
        results_dict[col] = {"stat": stat, "p": p, "dof": dof}
    
    result = pd.DataFrame(results_dict).T
    result['correlated'] = False
    result.loc[result['p'] < confidence_lvl, 'correlated'] = True
        
    return result


def find_boolean_features(X):
    """
    Args:
        X (pd.DataFrame) - input dataset
    Returns:
        bool_features (list): column names containing maximum 2 unique values
    """
    bool_features = []
    for col in X.columns:
        if len(X[col].unique()) <= 2:
            bool_features.append(col)
    return bool_features


def drop_correlated_features(X, threshold=0.9):
    # Create correlation matrix
    corr_matrix = X.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find features with correlation greater than threshold
    correlated_features = [column for column in upper.columns if any(upper[column] > threshold)]

    # Drop features 
    X = X.drop(correlated_features, axis=1)
    return X, correlated_features


def remove_outliers(X, cols):
    for col in cols:
        feature_std = X[col].std()
        feature_mean = X[col].mean()
        X.loc[X[col] > feature_mean + 3 * feature_std, col] = np.nan
        X.loc[X[col] < feature_mean - 3 * feature_std, col] = np.nan
    return X


def find_ordinal_features(X):
    ordinal_features = []
    cat_features = X.select_dtypes(include='object').columns
    for col in cat_features:
        if len(X[col].dropna().unique()) > 2:
            ordinal_features.append(col)
    return ordinal_features
