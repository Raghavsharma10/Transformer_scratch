def lx(mt, x):
    """ lx : Returns the number of survivors at begining of age x """    
    if x < len(mt.lx):
        return mt.lx[x]
    else:
        return 0