def bruggeman_refractive(m, mix):
    """Bruggeman EMA for the refractive index.

    For instructions, see mg_refractive in this module, except this routine
    only works for two components.
    """
    f1 = mix[0]/sum(mix)
    f2 = mix[1]/sum(mix)
    e1 = m[0]**2
    e2 = m[1]**2
    a = -2*(f1+f2)
    b = (2*f1*e1 - f1*e2 + 2*f2*e2 - f2*e1)
    c = (f1+f2)*e1*e2
    e_eff = (-b - np.sqrt(b**2-4*a*c))/(2*a)
    return np.sqrt(e_eff)