def _kvmatrix2d(km,vm):
    '''
        
        km = [[[1], [3]], [[1, 2], [3, 'a']], [[1, 2, 22]]]
        show_kmatrix(km)
        vm = [[[222]], ['b']]
        show_vmatrix(vm)
        
        d = _kvmatrix2d(km,vm)
    '''
    d = {}
    kmwfs = get_kmwfs(km)
    vmwfs = elel.get_wfs(vm)
    lngth = vmwfs.__len__()
    for i in range(0,lngth):
        value = elel.getitem_via_pathlist(vm,vmwfs[i])
        cond = elel.is_leaf(value)
        if(cond):
            _setitem_via_pathlist(d,kmwfs[i],value)
        else:
            _setdefault_via_pathlist(d,kmwfs[i])
    return(d)