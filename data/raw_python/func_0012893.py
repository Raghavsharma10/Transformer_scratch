def nocomment(astr, com='!'):
    """
    just like the comment in python.
    removes any text after the phrase 'com'
    """
    alist = astr.splitlines()
    for i in range(len(alist)):
        element = alist[i]
        pnt = element.find(com)
        if pnt != -1:
            alist[i] = element[:pnt]
    return '\n'.join(alist)