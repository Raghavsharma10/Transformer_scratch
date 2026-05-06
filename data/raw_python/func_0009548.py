def angleOfView(XY, shape=None, a=None, f=None, D=None, center=None):
    '''
    Another vignetting equation from:
    M. Koentges, M. Siebert, and D. Hinken, "Quantitative analysis of PV-modules by electroluminescence images for quality control"
        2009
    f --> Focal length
    D --> Diameter of the aperture
        BOTH, D AND f NEED TO HAVE SAME UNIT [PX, mm ...]
    a --> Angular aperture
    
    center -> optical center [y,x]
    '''
    if a is None:
        assert f is not None and D is not None
        #https://en.wikipedia.org/wiki/Angular_aperture
        a = 2*np.arctan2(D/2,f)
    
    x,y = XY

    try:
        c0,c1 = center
    except:
        s0,s1 = shape
        c0,c1 = s0/2, s1/2

    rx = (x-c0)**2
    ry = (y-c1)**2  

    return  1 / (1+np.tan(a)*((rx+ry)/c0))**0.5