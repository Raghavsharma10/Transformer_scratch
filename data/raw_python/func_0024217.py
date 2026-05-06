def safe_print(ustring, errors='replace', **kwargs):
    """ Safely print a unicode string """
    encoding = sys.stdout.encoding or 'utf-8'
    if sys.version_info[0] == 3:
        print(ustring, **kwargs)
    else:
        bytestr = ustring.encode(encoding, errors=errors)
        print(bytestr, **kwargs)