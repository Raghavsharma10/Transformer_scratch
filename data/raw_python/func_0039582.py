def cosi_posterior(vsini_dist,veq_dist,vgrid=None,npts=100,vgrid_pts=1000):
    """returns posterior of cosI given dists for vsini and veq (incorporates unc. in vsini)
    """
    if vgrid is None:
        vgrid = np.linspace(min(veq_dist.ppf(0.001),vsini_dist.ppf(0.001)),
                            max(veq_dist.ppf(0.999),vsini_dist.ppf(0.999)),
                            vgrid_pts)
        logging.debug('vgrid: {} pts, {} to {}'.format(vgrid_pts,
                                                       vgrid[0],
                                                       vgrid[-1]))
        #vgrid = np.linspace(vsini_dist.ppf(0.005),vsini_dist.ppf(0.995),vgrid_pts)
    cs = np.linspace(0,1,npts)
    Ls = cs*0
    for i,c in enumerate(cs):
        Ls[i] = like_cosi(c,vsini_dist,veq_dist,vgrid=vgrid)
    if np.isnan(Ls[-1]): #hack to prevent nan when cos=1
        Ls[-1] = Ls[-2]
    Ls /= np.trapz(Ls,cs)
    return cs,Ls