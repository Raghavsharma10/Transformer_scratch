def angleOfView2(x,y, b, x0=None,y0=None):
    '''
    Corrected AngleOfView equation by Koentges (via mail from 14/02/2017)
    b --> distance between the camera and the module in m
    x0 --> viewable with in the module plane of the camera in m
    y0 --> viewable height in the module plane of the camera in m
    x,y --> pixel position [m] from top left
    '''
    if x0 is None:
        x0 = x[-1,-1]
    if y0 is None:
        y0 = y[-1,-1]    
    return np.cos( np.arctan( np.sqrt(
                    ( (x-x0/2)**2+(y-y0/2)**2 ) ) /b  ) )