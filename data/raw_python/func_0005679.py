def escape_keywords(arr):
    """append _ to all python keywords"""
    for i in arr:
        i = i if i not in kwlist else i + '_'
        i = i if '-' not in i else i.replace('-', '_')
        yield i