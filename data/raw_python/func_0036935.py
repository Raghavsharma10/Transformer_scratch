def show_vmatrix(vm):
    '''
        d = {1: {2: {22: 222}}, 3: {'a': 'b'}}
        vm = [[[222]], ['b']]
        show_vmatrix(vm)
    '''
    unhandled = vm
    while(unhandled.__len__()>0):
        next_unhandled = []
        for i in range(0,unhandled.__len__()):
            ele = unhandled[i]
            print(ele)
            cond = elel.is_leaf(ele)
            if(cond):
                pass
            else:
                children = ele[0]
                next_unhandled.append(children)
        unhandled = next_unhandled