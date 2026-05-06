def intinlist(lst):
    """test if int in list"""
    for item in lst:
        try:
            item = int(item)
            return True
        except ValueError:
            pass
    return False