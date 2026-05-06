def _reorder_via_klist(d,nkl,**kwargs):
    '''
        d = {'scheme': 'http', 'path': '/index.php', 'params': 'params', 'query': 'username=query', 'fragment': 'frag', 'username': '', 'password': '', 'hostname': 'www.baidu.com', 'port': ''}
        pobj(d)
        nkl = ['scheme', 'username', 'password', 'hostname', 'port', 'path', 'params', 'query', 'fragment']
        pobj(_reorder_via_klist(d,nkl))
    '''
    if('deepcopy' in kwargs):
        deepcopy = kwargs['deepcopy']
    else:
        deepcopy = True
    if(deepcopy):
        d = copy.deepcopy(d)
    else:
        pass
    nd = {}
    lngth = nkl.__len__()
    for i in range(0,lngth):
        k = nkl[i]
        nd[k] = d[k]
    return(nd)