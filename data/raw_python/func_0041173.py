def fveq(z,R,dR,P,dP):
    """z=veq in km/s.  
    """

    R *= 2*pi*RSUN
    dR *= 2*pi*RSUN
    P *= DAY
    dP *= DAY

    exp1 = -P**2/(2*dP**2) - R**2/(2*dR**2)
    exp2 = (dR**2*P + dP**2*R*(z*1e5))**2/(2*dP**2*dR**2*(dR**2 + dP**2*(z*1e5)**2))

    nonexp_term = 2*dP*dR*np.sqrt(dR**2 + dP**2*(z*1e5)**2)

    return 1e5/(4*pi*(dR**2 + dP**2*(z*1e5)**2)**(3/2))*np.exp(exp1 + exp2) *\
        (dR**2 * P*np.sqrt(2*pi) + 
         dP**2 * np.sqrt(2*pi)*R*(z*1e5) + 
         nonexp_term * np.exp(-exp2) +
         np.sqrt(2*pi)*(dR**2*P + dP**2*R*(z*1e5))*erf((dR**2*P + dP**2*R*(z*1e5)) *
                                                       (np.sqrt(2)*dP*dR*
                                                        np.sqrt(dR**2 + dP**2*(z*1e5)**2))))