def get_vndmat_attr(d,keypath,attr,**kwargs):
    '''
        get_vndmat_attr(d,['x'],'lsib_path',path2keypath=True)
        get_vndmat_attr(d,['t'],'lsib_path',path2keypath=True)
        get_vndmat_attr(d,['u'],'lsib_path',path2keypath=True)
        get_vndmat_attr(d,['y'],'lsib_path',path2keypath=True)
    '''
    kt,vn = _d2kvmatrix(d)
    kdmat = _scankm(kt)
    ltree = elel.ListTree(vn)
    vndmat = ltree.desc
    loc = get_kdmat_loc(kdmat,keypath)
    rslt = vndmat[loc[0]][loc[1]][attr]
    if(rslt == None):
        pass
    elif(elel.is_matrix(rslt,mode='loose')):
        if('path2loc' in kwargs):
            rslt = elel.array_map(rslt,ltree.path2loc)
        else:
            pass
        if('path2keypath' in kwargs):
            nlocs = elel.array_map(rslt,ltree.path2loc)
            def cond_func(ele,kdmat):
                return(kdmat[ele[0]][ele[1]]['path'])
            rslt = elel.array_map(nlocs,cond_func,kdmat)
        else:
            pass        
    else:
        if('path2loc' in kwargs):
            rslt = ltree.path2loc(rslt)
        else:
            pass
        if('path2keypath' in kwargs):
            nloc = ltree.path2loc(rslt)
            rslt = kdmat[nloc[0]][nloc[1]]['path']
        else:
            pass
    return(rslt)