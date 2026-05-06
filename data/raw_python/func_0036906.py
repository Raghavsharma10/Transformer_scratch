def d2kvlist(d):
    '''
        d = {'GPSImgDirectionRef': 'M', 'GPSVersionID': b'\x02\x03\x00\x00', 'GPSImgDirection': (21900, 100)}
        pobj(d)
        kl,vl = d2kvlist(d)
        pobj(kl)
        pobj(vl)
    '''
    kl = list(d.keys())
    vl = list(d.values())
    return((kl,vl))