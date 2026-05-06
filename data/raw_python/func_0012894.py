def idf2txt(txt):
    """convert the idf text to a simple text"""
    astr = nocomment(txt)
    objs = astr.split(';')
    objs = [obj.split(',') for obj in objs]
    objs = [[line.strip() for line in obj] for obj in objs]
    objs = [[_tofloat(line) for line in obj] for obj in objs]
    objs = [tuple(obj) for obj in objs]
    objs.sort()

    lst = []
    for obj in objs:
        for field in obj[:-1]:
            lst.append('%s,' % (field, ))
        lst.append('%s;\n' % (obj[-1], ))

    return '\n'.join(lst)