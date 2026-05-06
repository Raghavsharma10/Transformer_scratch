def _diff_internal(d1,d2):
    '''
        d1 = {'a':'x','b':'y','c':'z'}
        d2 = {'a':'x','b':'u','d':'v'}
        _diff_internal(d1,d2)
        _diff_internald2,d1)
    '''
    same =[]
    kdiff =[]
    vdiff = []
    for key in d1:
        value = d1[key]
        if(key in d2):
            if(value == d2[key]):
                same.append(key)
            else:
                vdiff.append(key)
        else:
            kdiff.append(key)
    return({'same':same,'kdiff':kdiff,'vdiff':vdiff})