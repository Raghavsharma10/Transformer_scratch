def iddversiontuple(afile):
    """given the idd file or filehandle, return the version handle"""
    def versiontuple(vers):
        """version tuple"""
        return tuple([int(num) for num in vers.split(".")])
    try:
        fhandle = open(afile, 'rb')
    except TypeError:
        fhandle = afile
    line1 = fhandle.readline()
    try:
        line1 = line1.decode('ISO-8859-2')
    except AttributeError:
        pass
    line = line1.strip()
    if line1 == '':
        return (0,)
    vers = line.split()[-1]
    return versiontuple(vers)