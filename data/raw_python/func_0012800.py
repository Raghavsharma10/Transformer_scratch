def myreplace(astr, thefind, thereplace):
    """in string astr replace all occurences of thefind with thereplace"""
    alist = astr.split(thefind)
    new_s = alist.split(thereplace)
    return new_s