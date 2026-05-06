def veq_samples(R_dist,Prot_dist,N=1e4,alpha=0.23,l0=20,sigl=20):
    """Source for diff rot
    """
    ls = stats.norm(l0,sigl).rvs(N)
    Prots = Prot_dist.rvs(N)
    Prots *= diff_Prot_factor(ls,alpha)
    return R_dist.rvs(N)*2*np.pi*RSUN/(Prots*DAY)/1e5