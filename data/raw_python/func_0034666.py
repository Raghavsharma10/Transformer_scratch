def _sortByModified(a, b):
    """Sort function for object by modified date"""
    if a.modified < b.modified:
        return 1
    elif a.modified > b.modified:
        return -1
    else:
        return 0