def towgs84(E, N, pkm=False, presentation=None):
    """
    Convert coordintes from TWD97 to WGS84

    The east and north coordinates should be in meters and in float
    pkm true for Penghu, Kinmen and Matsu area
    You can specify one of the following presentations of the returned values:
        dms - A tuple with degrees (int), minutes (int) and seconds (float)
        dmsstr - [+/-]DDD°MMM'DDD.DDDDD" (unicode)
        mindec - A tuple with degrees (int) and minutes (float)
        mindecstr - [+/-]DDD°MMM.MMMMM' (unicode)
        (default)degdec - DDD.DDDDD (float)
    """

    _lng0 = lng0pkm if pkm else lng0

    E /= 1000.0
    N /= 1000.0
    epsilon = (N-N0) / (k0*A)
    eta = (E-E0) / (k0*A)

    epsilonp = epsilon - beta1*sin(2*1*epsilon)*cosh(2*1*eta) - \
                         beta2*sin(2*2*epsilon)*cosh(2*2*eta) - \
                         beta3*sin(2*3*epsilon)*cosh(2*3*eta)
    etap = eta - beta1*cos(2*1*epsilon)*sinh(2*1*eta) - \
                 beta2*cos(2*2*epsilon)*sinh(2*2*eta) - \
                 beta3*cos(2*3*epsilon)*sinh(2*3*eta)
    sigmap = 1 - 2*1*beta1*cos(2*1*epsilon)*cosh(2*1*eta) - \
                 2*2*beta2*cos(2*2*epsilon)*cosh(2*2*eta) - \
                 2*3*beta3*cos(2*3*epsilon)*cosh(2*3*eta)
    taup = 2*1*beta1*sin(2*1*epsilon)*sinh(2*1*eta) + \
           2*2*beta2*sin(2*2*epsilon)*sinh(2*2*eta) + \
           2*3*beta3*sin(2*3*epsilon)*sinh(2*3*eta)

    chi = asin(sin(epsilonp) / cosh(etap))

    latitude = chi + delta1*sin(2*1*chi) + \
                     delta2*sin(2*2*chi) + \
                     delta3*sin(2*3*chi)

    longitude = _lng0 + atan(sinh(etap) / cos(epsilonp))

    func = None
    presentation = 'to%s' % presentation if presentation else None
    if presentation in presentations:
        func = getattr(sys.modules[__name__], presentation)

    if func and func != 'todegdec':
        return func(degrees(latitude)), func(degrees(longitude))

    return (degrees(latitude), degrees(longitude))