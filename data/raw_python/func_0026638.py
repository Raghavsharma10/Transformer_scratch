def fromwgs84(lat, lng, pkm=False):
    """
    Convert coordintes from WGS84 to TWD97

    pkm true for Penghu, Kinmen and Matsu area
    The latitude and longitude can be in the following formats:
        [+/-]DDD°MMM'SSS.SSSS" (unicode)
        [+/-]DDD°MMM.MMMM' (unicode)
        [+/-]DDD.DDDDD (string, unicode or float)
    The returned coordinates are in meters
    """

    _lng0 = lng0pkm if pkm else lng0

    lat = radians(todegdec(lat))
    lng = radians(todegdec(lng))

    t = sinh((atanh(sin(lat)) - 2*pow(n,0.5)/(1+n)*atanh(2*pow(n,0.5)/(1+n)*sin(lat))))
    epsilonp = atan(t/cos(lng-_lng0))
    etap = atan(sin(lng-_lng0) / pow(1+t*t, 0.5))

    E = E0 + k0*A*(etap + alpha1*cos(2*1*epsilonp)*sinh(2*1*etap) + 
                          alpha2*cos(2*2*epsilonp)*sinh(2*2*etap) +
                          alpha3*cos(2*3*epsilonp)*sinh(2*3*etap))
    N = N0 + k0*A*(epsilonp + alpha1*sin(2*1*epsilonp)*cosh(2*1*etap) +
                              alpha2*sin(2*2*epsilonp)*cosh(2*2*etap) +
                              alpha3*sin(2*3*epsilonp)*cosh(2*3*etap))

    return E*1000, N*1000