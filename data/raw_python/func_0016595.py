def normcdf(x, mu, sigma):
    """
    Describes the probability that a real-valued random variable X with a given
    probability distribution will be found at a value less than or equal to X
    in a normal distribution.
    
    http://en.wikipedia.org/wiki/Cumulative_distribution_function
    """
    t = x-mu
    y = 0.5*erfcc(-t/(sigma*math.sqrt(2.0)))
    if y > 1.0:
        y = 1.0
    return y