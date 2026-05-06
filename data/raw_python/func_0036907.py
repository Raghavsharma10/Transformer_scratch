def klviavl(d,vl):
    '''
        must be 1:1 map
    '''
    dkl,dvl = d2kvlist(d)
    kl = []
    for i in range(vl.__len__()):
        v = vl[i]
        index = dvl.index(v)
        kl.append(dkl[index])
    return(kl)