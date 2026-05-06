def is_dicom(filename):
    '''returns Boolean of whether the given file has the DICOM magic number'''
    try:
        with open(filename) as f:
            d = f.read(132)
            return d[128:132]=="DICM"
    except:
        return False