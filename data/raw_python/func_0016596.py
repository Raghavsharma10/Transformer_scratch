def normpdf(x, mu, sigma):
    """
    Describes the relative likelihood that a real-valued random variable X will
    take on a given value.
    
    http://en.wikipedia.org/wiki/Probability_density_function
    """
    u = (x-mu)/abs(sigma)
    y = (1/(math.sqrt(2*pi)*abs(sigma)))*math.exp(-u*u/2)
    return y