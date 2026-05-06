def generate_veq(R=1.3, dR=0.1, Prot=6, dProt=0.1,nsamples=1e4,plot=False,
                 R_samples=None,Prot_samples=None):
    """ Returns the mean and std equatorial velocity given R,dR,Prot,dProt

    Assumes all distributions are normal.  This will be used mainly for
    testing purposes; I can use MC-generated v_eq distributions when we go for real.
    """
    if R_samples is None:
        R_samples = R*(1 + rand.normal(size=nsamples)*dR)
    else:
        inds = rand.randint(len(R_samples),size=nsamples)
        R_samples = R_samples[inds]

    if Prot_samples is None:
        Prot_samples = Prot*(1 + rand.normal(size=nsamples)*dProt)
    else:
        inds = rand.randint(len(Prot_samples),size=nsamples)
        Prot_samples = Prot_samples[inds]

    veq_samples = 2*np.pi*R_samples*RSUN/(Prot_samples*DAY)/1e5
    
    if plot:
        plt.hist(veq_samples,histtype='step',color='k',bins=50,normed=True)
        d = stats.norm(scale=veq_samples.std(),loc=veq_samples.mean())
        vs = np.linspace(veq_samples.min(),veq_samples.max(),1e4)
        plt.plot(vs,d.pdf(vs),'r')
    
    return veq_samples.mean(),veq_samples.std()