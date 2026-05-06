def roc_auc_xlim(x_bla, y_bla, xlim=0.1):
    """
    Computes the ROC Area Under Curve until a certain FPR value.

    Parameters
    ----------
    fg_vals : array_like
        list of values for positive set

    bg_vals : array_like
        list of values for negative set

    xlim : float, optional
        FPR value
    
    Returns
    -------
    score : float
        ROC AUC score
    """
    x = x_bla[:]
    y = y_bla[:]

    x.sort()
    y.sort()

    u = {}
    for i in x + y:
        u[i] = 1

    vals = sorted(u.keys())
    
    len_x = float(len(x))
    len_y = float(len(y))
    
    new_x = []
    new_y = []
    
    x_p = 0
    y_p = 0
    for val in vals[::-1]:
        while len(x) > 0 and x[-1] >= val:
            x.pop()
            x_p += 1
        while len(y) > 0 and y[-1] >= val:
            y.pop()
            y_p += 1
        new_y.append((len_x - x_p) / len_x)
        new_x.append((len_y - y_p) / len_y)
    
    #print new_x
    #print new_y
    new_x = 1 - np.array(new_x)
    new_y = 1 - np.array(new_y)
    #plot(new_x, new_y)
    #show()

    x = new_x
    y = new_y

    if len(x) != len(y):
        raise ValueError("Unequal!")

    if not xlim:
        xlim = 1.0

    auc = 0.0
    bla = zip(stats.rankdata(x), range(len(x)))
    bla = sorted(bla, key=lambda x: x[1])
    
    prev_x = x[bla[0][1]]
    prev_y = y[bla[0][1]]
    index = 1

    while index < len(bla) and x[bla[index][1]] <= xlim:

        _, i = bla[index]
        
        auc += y[i] * (x[i] - prev_x) - ((x[i] - prev_x) * (y[i] - prev_y) / 2.0)
        prev_x = x[i]
        prev_y = y[i]
        index += 1
    
    if index < len(bla):
        (rank, i) = bla[index]
        auc += prev_y * (xlim - prev_x) + ((y[i] - prev_y)/(x[i] - prev_x) * (xlim -prev_x) * (xlim - prev_x)/2)
 
    return auc