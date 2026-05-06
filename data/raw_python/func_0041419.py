def hash(filename):
    '''returns string of MD5 hash of given filename'''
    buffer_size = 10*1024*1024
    m = hashlib.md5()
    with open(filename) as f:
        buff = f.read(buffer_size)
        while len(buff)>0:
            m.update(buff)
            buff = f.read(buffer_size)
    dig = m.digest()
    return ''.join(['%x' % ord(x) for x in dig])