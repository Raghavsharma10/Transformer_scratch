def sample_posterior(x,post,nsamples=1):
    """ Returns nsamples from a tabulated posterior (not necessarily normalized)
    """
    cdf = post.cumsum()
    cdf /= cdf.max()
    u = rand.random(size=nsamples)
    inds = np.digitize(u,cdf)
    return x[inds]