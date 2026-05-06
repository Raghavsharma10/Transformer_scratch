def cleanupversion(ver):
    """massage the version number so it matches the format of install folder"""
    lst = ver.split(".")
    if len(lst) == 1:
        lst.extend(['0', '0'])
    elif len(lst) == 2:
        lst.extend(['0'])
    elif len(lst) > 2:
        lst = lst[:3]
    lst[2] = '0' # ensure the 3rd number is 0    
    cleanver = '.'.join(lst)
    return cleanver