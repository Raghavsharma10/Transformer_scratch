def _keys_via_value(d,value,**kwargs):
    '''
        d = {
         'x':
              {
               'x2': 'x22',
               'x1': 'x11'
              },
         'y':
              {
               'y1': 'v1',
               'y2':
                     {
                      'y4': 'v4',
                      'y3': 'v3',
                     },
               'xx': 
                    {
                        'x2': 'x22',
                        'x1': 'x11'
                  }
              },
         't': 20,
         'u':
              {
               'u1': 20
              }
        }
    '''
    km,vm = _d2kvmatrix(d)
    rvmat = _get_rvmat(d)
    depth = rvmat.__len__()
    ##
    #print(km)
    ##
    kdmat = _scankm(km)
    if('leaf_only' in kwargs):
        leaf_only = kwargs['leaf_only']
    else:
        leaf_only = False
    if('non_leaf_only' in kwargs):
        non_leaf_only = kwargs['non_leaf_only']
    else:
        non_leaf_only = False
    if('from_lv' in kwargs):
        from_lv = kwargs['from_lv']
    else:
        from_lv = 1
    if('to_lv' in kwargs):
        to_lv = kwargs['to_lv']
    else:
        if('from_lv' in kwargs):
            to_lv = from_lv
        else:
            to_lv = depth
    rslt = []
    for i in range(from_lv,to_lv):
        rvlevel = rvmat[i]
        for j in range(0,rvlevel.__len__()):
            v = rvlevel[j]
            cond1 = (v == value)
            if(leaf_only == True):
                cond2 = (kdmat[i][j]['leaf'] == True)
            elif(non_leaf_only == True):
                cond2 = (kdmat[i][j]['leaf'] == False)
            else:
                cond2 = True
            cond = (cond1 & cond2)
            if(cond):
                rslt.append(kdmat[i][j]['path'])
            else:
                pass
    return(rslt)