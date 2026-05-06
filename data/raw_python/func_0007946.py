def _merge(listA, listB):
    """ Merges two list of objects removing
    repetitions. 
    
    """
    listA = [x.id for x in listA]
    listB = [x.id for x in listB]
    listA.extend(listB)
    set_ = set(listA)
    return list(set_)