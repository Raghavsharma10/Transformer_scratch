def fisher(x,k):
    """Fisher distribution 
    """
    return k/(2*np.sinh(k)) * np.exp(k*np.cos(x))*np.sin(x)