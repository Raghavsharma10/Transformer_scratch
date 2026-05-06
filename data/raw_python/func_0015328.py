def compare(ver1, ver2):
    '''Version comparing, returns 0 if are the same,
    returns >0 if first is bigger, <0 if first is smaller,
    excepts valid version'''
    if ver1 == ver2:
        return 0
    ver1 = _cut(ver1)
    ver2 = _cut(ver2)
    # magic multiplier
    m = 1
    # if the first one is shorter, replace them
    if len(ver1) < len(ver2):
        ver1, ver2 = ver2, ver1
        # and reverse magic multiplier
        m = -1
    # compare all items that both have
    for i, part in enumerate(ver2):
        if ver1[i] > ver2[i]:
            return m * 1
        if ver1[i] < ver2[i]:
            return m * -1
    # if the first "extra" item is not negative, it's bigger
    if ver1[len(ver2)] >= 0:
        return m * 1
    else:
        return m * -1