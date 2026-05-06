def readfile(filename):
    """readfile"""
    fhandle = open(filename, 'rb')
    data = fhandle.read()
    try:
        data = data.decode('ISO-8859-2')
    except AttributeError:  
        pass
    fhandle.close()
    return data