def withFile(file, func, mode='r', expand=False):
    """Pass `file` to `func` and ensure the file is closed afterwards. If
       `file` is a string, open according to `mode`; if `expand` is true also
       expand user and vars.
    """
    file = _normalizeToFile(file, mode=mode, expand=expand)
    try:      return func(file)
    finally:  file.close()