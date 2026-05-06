def offsetMeshgrid(offset, grid, shape):
    '''
    Imagine you have cell averages [grid] on an image.
    the top-left position of [grid] within the image 
    can be variable [offset]
    
    offset(x,y) 
        e.g.(0,0) if no offset
    grid(nx,ny) resolution of smaller grid
    shape(x,y) -> output shape 
    
    returns meshgrid to be used to upscale [grid] to [shape] resolution
    '''    
    g0,g1 = grid
    s0,s1 = shape
    o0, o1 = offset
    #rescale to small grid:
    o0 = - o0/ s0 * (g0-1)
    o1 = - o1/ s1 * (g1-1)

    xx,yy = np.meshgrid(np.linspace(o1, o1+g1-1, s1),
                        np.linspace(o0,o0+g0-1,  s0))
    return yy,xx