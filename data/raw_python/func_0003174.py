def mg_refractive(m, mix):
    """Maxwell-Garnett EMA for the refractive index.

    Args:
       m: Tuple of the complex refractive indices of the media.
       mix: Tuple of the volume fractions of the media, len(mix)==len(m)
            (if sum(mix)!=1, these are taken relative to sum(mix))

    Returns:
       The Maxwell-Garnett approximation for the complex refractive index of 
       the effective medium

    If len(m)==2, the first element is taken as the matrix and the second as 
    the inclusion. If len(m)>2, the media are mixed recursively so that the 
    last element is used as the inclusion and the second to last as the 
    matrix, then this mixture is used as the last element on the next 
    iteration, and so on.
    """

    if len(m) == 2:
        cF = float(mix[1]) / (mix[0]+mix[1]) * \
            (m[1]**2-m[0]**2) / (m[1]**2+2*m[0]**2)
        er = m[0]**2 * (1.0+2.0*cF) / (1.0-cF)
        m = np.sqrt(er)
    else:
        m_last = mg_refractive(m[-2:], mix[-2:])
        mix_last = mix[-2] + mix[-1]
        m = mg_refractive(m[:-2] + (m_last,), mix[:-2] + (mix_last,))
    return m