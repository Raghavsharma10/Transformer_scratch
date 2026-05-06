def mkpad(items):
    '''
    Find the length of the longest element of a list. Return that value + two.
    '''
    pad = 0
    stritems = [str(e) for e in items]  # cast list to strings
    for e in stritems:
        index = stritems.index(e)
        if len(stritems[index]) > pad:
            pad = len(stritems[index])
    pad += 2
    return pad