def safestr(str_):
    ''' get back an alphanumeric only version of source '''
    str_ = str_ or ""
    return "".join(x for x in str_ if x.isalnum())