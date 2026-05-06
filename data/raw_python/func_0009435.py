def polyfit2d(x, y, z, order=3 #bounds=None
              ):
    '''
    fit unstructured data 
    '''
    ncols = (order + 1)**2
    G = np.zeros((x.size, ncols))
    ij = itertools.product(list(range(order+1)), list(range(order+1)))
    for k, (i,j) in enumerate(ij):
        G[:,k] = x**i * y**j
    m = np.linalg.lstsq(G, z)[0]
    return m