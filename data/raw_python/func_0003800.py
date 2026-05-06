def _opbend_transform_mean(rs, fn_low, deriv=0):
    """Compute the mean of the 3 opbends
    """
    v = 0.0
    d = np.zeros((4,3), float)
    dd = np.zeros((4,3,4,3), float)
    #loop over the 3 cyclic permutations
    for p in np.array([[0,1,2], [2,0,1], [1,2,0]]):
        opbend = _opbend_transform([rs[p[0]], rs[p[1]], rs[p[2]], rs[3]], fn_low, deriv)
        v += opbend[0]/3
        index0 = np.where(p==0)[0][0] #index0 is the index of the 0th atom (rs[0])
        index1 = np.where(p==1)[0][0]
        index2 = np.where(p==2)[0][0]
        index3 = 3
        if deriv>0:
            d[0] += opbend[1][index0]/3
            d[1] += opbend[1][index1]/3
            d[2] += opbend[1][index2]/3
            d[3] += opbend[1][index3]/3
        if deriv>1:
            dd[0, :, 0, :] += opbend[2][index0, :, index0, :]/3
            dd[0, :, 1, :] += opbend[2][index0, :, index1, :]/3
            dd[0, :, 2, :] += opbend[2][index0, :, index2, :]/3
            dd[0, :, 3, :] += opbend[2][index0, :, index3, :]/3

            dd[1, :, 0, :] += opbend[2][index1, :, index0, :]/3
            dd[1, :, 1, :] += opbend[2][index1, :, index1, :]/3
            dd[1, :, 2, :] += opbend[2][index1, :, index2, :]/3
            dd[1, :, 3, :] += opbend[2][index1, :, index3, :]/3

            dd[2, :, 0, :] += opbend[2][index2, :, index0, :]/3
            dd[2, :, 1, :] += opbend[2][index2, :, index1, :]/3
            dd[2, :, 2, :] += opbend[2][index2, :, index2, :]/3
            dd[2, :, 3, :] += opbend[2][index2, :, index3, :]/3

            dd[3, :, 0, :] += opbend[2][index3, :, index0, :]/3
            dd[3, :, 1, :] += opbend[2][index3, :, index1, :]/3
            dd[3, :, 2, :] += opbend[2][index3, :, index2, :]/3
            dd[3, :, 3, :] += opbend[2][index3, :, index3, :]/3
    if deriv==0:
        return v,
    elif deriv==1:
        return v, d
    elif deriv==2:
        return v, d, dd
    else:
        raise ValueError("deriv must be 0, 1 or 2.")