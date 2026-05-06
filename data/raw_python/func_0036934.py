def show_kmatrix(km):
    '''
        d = {1: {2: {22: 222}}, 3: {'a': 'b'}}
        km = [[[1], [3]], [[1, 2], [3, 'a']], [[1, 2, 22]]]
        show_kmatrix(km)
    '''
    rslt = []
    for i in range(0,km.__len__()):
        level = km[i]
        for j in range(0,level.__len__()):
            kpl = level[j]
            print(kpl)
            rslt.append(kpl)
    return(rslt)