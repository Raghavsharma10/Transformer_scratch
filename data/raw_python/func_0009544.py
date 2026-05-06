def navierStokes2d(u, v, p, dt, nt, rho, nu,  
                boundaryConditionUV, 
                boundardConditionP, nit=100):
    '''
    solves the Navier-Stokes equation for incompressible flow
    one a regular 2d grid
    
    u,v,p --> initial velocity(u,v) and pressure(p) maps
    
    dt --> time step
    nt --> number of time steps to caluclate
    
    rho, nu --> material constants
    
    nit --> number of iteration to solve the pressure field
    '''
    #next u, v, p maps:
    un = np.empty_like(u)
    vn = np.empty_like(v)
    pn = np.empty_like(p)
    #poisson equation ==> laplace term = b[source term]
    b = np.zeros_like(p)

    ny,nx = p.shape
    #cell size:
    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)
    
    #next time step:
    for _ in range(nt):
        un[:] = u
        vn[:] = v
        #pressure
        _buildB(b, rho, dt, u, v, dx, dy)
        for _ in range(nit):
            _pressurePoisson(p, pn, dx, dy, b)
            boundardConditionP(p)
        #UV
        _calcUV(u, v, un, p,vn,  dt, dx, dy, rho, nu)
        boundaryConditionUV(u,v)

    return u, v, p