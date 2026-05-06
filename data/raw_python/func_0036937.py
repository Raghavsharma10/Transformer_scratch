def _get_rvmat(d):
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
                      'y3': 'v3'
                     }
              },
         't': 20,
         'u':
              {
               'u1': 'u2'
              }
        }
        
        
    '''
    km,vm = _d2kvmatrix(d)
    def map_func(ele,indexc,indexr):
        return(_getitem_via_pathlist(d,ele))
    rvmat = elel.matrix_map(km,map_func)
    rvmat = elel.prepend(rvmat,[])
    return(rvmat)