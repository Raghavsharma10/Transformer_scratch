def polyfit2dGrid(arr, mask=None, order=3, replace_all=False,
                  copy=True, outgrid=None):
    '''
    replace all masked values with polynomial fitted ones
    '''
    s0,s1 = arr.shape
    if mask is None:
        if outgrid is None:
            y,x = np.mgrid[:float(s0),:float(s1)]
            p = polyfit2d(x.flatten(),y.flatten(),arr.flatten(),order)
            return polyval2d(x,y, p, dtype=arr.dtype)
        mask = np.zeros_like(arr, dtype=bool)
    elif mask.sum() == 0 and not replace_all and outgrid is None:

        return arr
    valid = ~mask
    y,x = np.where(valid)
    z = arr[valid]
    p = polyfit2d(x,y,z,order)
    if outgrid is not None:
        yy,xx = outgrid
    else:
        if replace_all:
            yy,xx = np.mgrid[:float(s0),:float(s1)]
        else:
            yy,xx = np.where(mask)
    new = polyval2d(xx,yy, p, dtype=arr.dtype)
    
    if outgrid is not None or replace_all:
        return new
    
    if copy:
        arr = arr.copy()
    arr[mask] = new
    return arr