def flattencopy(lst):
    """flatten and return a copy of the list
    indefficient on large lists"""
    # modified from
    # http://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists-in-python
    thelist = copy.deepcopy(lst)
    list_is_nested = True
    while list_is_nested:  # outer loop
        keepchecking = False
        atemp = []
        for element in thelist:  # inner loop
            if isinstance(element, list):
                atemp.extend(element)
                keepchecking = True
            else:
                atemp.append(element)
        list_is_nested = keepchecking  # determine if outer loop exits
        thelist = atemp[:]
    return thelist