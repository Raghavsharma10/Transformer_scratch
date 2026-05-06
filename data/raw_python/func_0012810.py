def simplesurface(idf, bsd, deletebsd=True, setto000=False):
    """convert a bsd (buildingsurface:detailed) into a simple surface"""
    funcs = (wallexterior,
        walladiabatic,
        wallunderground,
        wallinterzone,
        roof,
        ceilingadiabatic,
        ceilinginterzone,
        floorgroundcontact,
        flooradiabatic,
        floorinterzone,)
    for func in funcs:
        surface = func(idf, bsd, deletebsd=deletebsd, setto000=setto000)
        if surface:
            return surface
    return None