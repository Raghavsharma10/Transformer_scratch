def standardize(Y,in_place=False):
    """
    standardize Y in a way that is robust to missing values
    in_place: create a copy or carry out inplace opreations?
    """
    if in_place:
        YY = Y
    else:
        YY = Y.copy()
    for i in range(YY.shape[1]):
        Iok = ~SP.isnan(YY[:,i])
        Ym = YY[Iok,i].mean()
        YY[:,i]-=Ym
        Ys = YY[Iok,i].std()
        YY[:,i]/=Ys
    return YY