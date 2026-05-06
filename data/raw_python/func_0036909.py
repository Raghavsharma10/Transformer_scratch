def _vksdesc(d):
    '''
        d = {'a':1,'b':2,'c':2,'d':4}
        desc = _vksdesc(d)
        pobj(desc)
    '''
    pt = copy.deepcopy(d)
    seqs_for_del =[]
    vset = set({})
    for k in pt:
        vset.add(pt[k])
    desc = {}
    for v in vset:
        desc[v] = []
    for k in pt:
        desc[pt[k]].append(k)
    return(desc)