def build_plane_arrays(x, y, qlist):
    """Build a 2-D array out of data taken in the same plane, for contour
    plotting.
    """
    if type(qlist) is not list:
        return_list = False
        qlist = [qlist]
    else:
        return_list = True
    xv = x[np.where(y==y[0])[0]]
    yv = y[np.where(x==x[0])[0]]
    qlistp = []
    for n in range(len(qlist)):
        qlistp.append(np.zeros((len(yv), len(xv))))
    for j in range(len(qlist)):
        for n in range(len(yv)):
            i = np.where(y==yv[n])[0]
            qlistp[j][n,:] = qlist[j][i]
    if not return_list:
        qlistp = qlistp[0]
    return xv, yv, qlistp