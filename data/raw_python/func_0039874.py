def tzname_in_python2(myfunc):
    """Change unicode output into bytestrings in Python 2

    tzname() API changed in Python 3. It used to return bytes, but was changed
    to unicode strings
    """
    def inner_func(*args, **kwargs):
        if PY3:
            return myfunc(*args, **kwargs)
        else:
            return myfunc(*args, **kwargs).encode()
    return inner_func