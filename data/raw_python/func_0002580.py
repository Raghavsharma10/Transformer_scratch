def dx(mt, x):
    """ Returns the number of dying at begining of age x """ 
    end_x_val = mt.lx.index(0)
    if x < end_x_val:  
        return mt.lx[x] - mt.lx[x + 1]
    else:
        return 0.0