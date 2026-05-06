def hash_str(string):
    '''returns string of MD5 hash of given string'''
    m = hashlib.md5()
    m.update(string)
    dig = m.digest()
    return ''.join(['%x' % ord(x) for x in dig])