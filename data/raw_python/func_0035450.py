def boxcox(X):
    """
    Gaussianize X using the Box-Cox transformation: [samples x phenotypes]

    - each phentoype is brought to a positive schale, by first subtracting the minimum value and adding 1.
    - Then each phenotype transformed by the boxcox transformation
    """
    X_transformed = sp.zeros_like(X)
    maxlog = sp.zeros(X.shape[1])
    for i in range(X.shape[1]):
        i_nan = sp.isnan(X[:,i])
        values = X[~i_nan,i]
        X_transformed[i_nan,i] = X[i_nan,i]
        X_transformed[~i_nan,i], maxlog[i] = st.boxcox(values-values.min()+1.0)
    return X_transformed, maxlog