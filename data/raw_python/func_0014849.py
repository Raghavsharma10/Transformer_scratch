def polydata_fromfile(f, self):
    """Use VtkData(<filename>)."""
    points = []
    data = dict(vertices=[], lines=[], polygons=[], triangle_strips=[])
    l = common._getline(f).decode('ascii')
    k,n,datatype = [s.strip().lower() for s in l.split(' ')]
    if k!='points':
        raise ValueError('expected points but got %s'%(repr(k)))
    n = int(n)
    assert datatype in ['bit','unsigned_char','char','unsigned_short','short','unsigned_int','int','unsigned_long','long','float','double'],repr(datatype)

    log.debug('\tgetting %s points'%n)
    while len(points) < 3*n:
        l = common._getline(f).decode('ascii')
        points += map(eval,l.split(' '))
    assert len(points)==3*n
    while 1:
        l = common._getline(f)
        if l is None:
            break
        l = l.decode('ascii')
        sl = l.split(' ')
        k = sl[0].strip().lower()
        if k not in ['vertices','lines','polygons','triangle_strips']:
            break
        assert len(sl)==3
        n = int(sl[1])
        size = int(sl[2])
        lst = []
        while len(lst) < size:
            l = common._getline(f).decode('ascii')
            lst += map(eval, l.split(' '))
        assert len(lst)==size
        lst2 = []
        j = 0
        for i in range(n):
            lst2.append(lst[j+1:j+lst[j]+1])
            j += lst[j]+1
        data[k] = lst2

    return PolyData(points,data['vertices'], data['lines'], data['polygons'], data['triangle_strips']), l.encode()