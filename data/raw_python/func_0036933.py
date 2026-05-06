def _d2kvmatrix(d):
    '''
        d = {1: 2, 3: {'a': 'b'}}
        km,vm = _d2kvmatrix(d)
        d = {1: {2:{22:222}}, 3: {'a': 'b'}}
        km,vm = _d2kvmatrix(d)
        ##
        km: 按照层次存储pathlist,层次从0开始，
        {
         1: 2,
         3:
            {
             'a': 'b'
            }
        }
        km[0] = [[1],[3]]
        km[1] = [[3,'a']]
        vm: vm比较特殊，不太好理解，请参照函数elel.get_wfs 和_kvmatrix2d
            vm的数组表示层次
        rvmat: 与km对应，存储key对应的value,不过对应层次使km的层次+1
    '''
    km = []
    vm = [list(d.values())]
    vm_history ={0:[0]}
    unhandled = [{'data':d,'kpl':[]}]
    while(unhandled.__len__()>0):
        next_unhandled = []
        keys_level = []
        next_vm_history = {}
        for i in range(0,unhandled.__len__()):
            data = unhandled[i]['data']
            kpl = unhandled[i]['kpl']
            values = list(data.values())
            _setitem_via_pathlist(vm,vm_history[i],values)
            vm_pl = vm_history[i]
            del vm_history[i]
            keys = data.keys()
            keys = elel.array_map(keys,_gen_sonpl,kpl)
            keys_level.extend(keys)
            for j in range(0,values.__len__()):
                v = values[j]
                cond = is_leaf(v)
                if(cond):
                    pass
                else:
                    kpl = copy.deepcopy(keys[j])
                    next_unhandled.append({'data':v,'kpl':kpl})
                    vpl = copy.deepcopy(vm_pl)
                    vpl.append(j)
                    next_vm_history[next_unhandled.__len__()-1] = vpl
        vm_history = next_vm_history
        km.append(keys_level)
        unhandled = next_unhandled
    vm = vm[0]
    return((km,vm))