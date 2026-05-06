def dtype_reduce(dtype, level=0, depth=0):
    """
    Try to reduce dtype up to a given level when it is possible

    dtype =  [ ('vertex',  [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]),
               ('normal',  [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]),
               ('color',   [('r', 'f4'), ('g', 'f4'), ('b', 'f4'),
                            ('a', 'f4')])]

    level 0: ['color,vertex,normal,', 10, 'float32']
    level 1: [['color', 4, 'float32']
              ['normal', 3, 'float32']
              ['vertex', 3, 'float32']]
    """
    dtype = np.dtype(dtype)
    fields = dtype.fields

    # No fields
    if fields is None:
        if len(dtype.shape):
            count = reduce(mul, dtype.shape)
        else:
            count = 1
        # size = dtype.itemsize / count
        if dtype.subdtype:
            name = str(dtype.subdtype[0])
        else:
            name = str(dtype)
        return ['', count, name]
    else:
        items = []
        name = ''
        # Get reduced fields
        for key, value in fields.items():
            l = dtype_reduce(value[0], level, depth + 1)
            if type(l[0]) is str:
                items.append([key, l[1], l[2]])
            else:
                items.append(l)
            name += key + ','

        # Check if we can reduce item list
        ctype = None
        count = 0
        for i, item in enumerate(items):
            # One item is a list, we cannot reduce
            if type(item[0]) is not str:
                return items
            else:
                if i == 0:
                    ctype = item[2]
                    count += item[1]
                else:
                    if item[2] != ctype:
                        return items
                    count += item[1]
        if depth >= level:
            return [name, count, ctype]
        else:
            return items