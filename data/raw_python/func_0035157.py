def MWColorMapping(maptype='jet', reverse=True):
    """Maps amino-acid molecular weights to colors. Otherwise, this
    function is identical to *KyteDoolittleColorMapping*
    """ 
    d = {'A':89,'R':174,'N':132,'D':133,'C':121,'Q':146,'E':147,\
         'G':75,'H':155,'I':131,'L':131,'K':146,'M':149,'F':165,\
         'P':115,'S':105,'T':119,'W':204,'Y':181,'V':117}
    
    aas = sorted(AA_TO_INDEX.keys())
    mws  = [d[aa] for aa in aas]
    if reverse:
        mws = [-1 * x for x in mws]
    mapper = pylab.cm.ScalarMappable(cmap=maptype)
    mapper.set_clim(min(mws), max(mws))
    mapping_d = {'*':'#000000'}
    for (aa, h) in zip(aas, mws):
        tup = mapper.to_rgba(h, bytes=True)
        (red, green, blue, alpha) = tup
        mapping_d[aa] = '#%02x%02x%02x' % (red, green, blue)
        assert len(mapping_d[aa]) == 7
    cmap = mapper.get_cmap()
    return (cmap, mapping_d, mapper)