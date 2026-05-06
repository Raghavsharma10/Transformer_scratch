def like_cosi(cosi,vsini_dist,veq_dist,vgrid=None):
    """likelihood of Data (vsini_dist, veq_dist) given cosi
    """
    sini = np.sqrt(1-cosi**2)
    def integrand(v):
        #return vsini_dist(v)*veq_dist(v/sini)
        return vsini_dist(v*sini)*veq_dist(v)
    if vgrid is None:
        return quad(integrand,0,np.inf)[0]
    else:
        return np.trapz(integrand(vgrid),vgrid)