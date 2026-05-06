def linePlot(img, x0, y0, x1, y1, resolution=None, order=3):
    '''
    returns [img] intensity values along line
    defined by [x0, y0, x1, y1]
    
    resolution ... number or data points to evaluate
    order ... interpolation precision
    '''
    if resolution is None:
        resolution = int( ((x1-x0)**2 + (y1-y0)**2 )**0.5 )
    
    if order == 0:
        x = np.linspace(x0, x1, resolution, dtype=int)
        y = np.linspace(y0, y1, resolution, dtype=int)
        return img[y, x]

    x = np.linspace(x0, x1, resolution)
    y = np.linspace(y0, y1, resolution)
    return map_coordinates(img, np.vstack((y,x)), order=order)