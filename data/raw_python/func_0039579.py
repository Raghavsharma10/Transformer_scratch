def cosi_pdf(z,k=1):
    """Equation (11) of Morton & Winn (2014)
    """
    return 2*k/(np.pi*np.sinh(k)) * quad(cosi_integrand,z,1,args=(k,z))[0]