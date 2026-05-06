def simplefenestration(idf, fsd, deletebsd=True, setto000=False):
    """convert a bsd (fenestrationsurface:detailed) into a simple 
    fenestrations"""
    funcs = (window,
        door,
        glazeddoor,)
    for func in funcs:
        fenestration = func(idf, fsd, deletebsd=deletebsd, setto000=setto000)
        if fenestration:
            return fenestration
    return None