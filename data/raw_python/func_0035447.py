def gaussianize(Y):
    """
    Gaussianize X: [samples x phenotypes]
    - each phentoype is converted to ranks and transformed back to normal using the inverse CDF
    """
    N,P = Y.shape

    YY=toRanks(Y)
    quantiles=(sp.arange(N)+0.5)/N
    gauss = st.norm.isf(quantiles)
    Y_gauss=sp.zeros((N,P))
    for i in range(P):
        Y_gauss[:,i] = gauss[YY[:,i]]
    Y_gauss *= -1
    return Y_gauss