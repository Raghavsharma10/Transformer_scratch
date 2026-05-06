def idfreader1(fname, iddfile, theidf, conv=True, commdct=None, block=None):
    """read idf file and return bunches"""
    versiontuple = iddversiontuple(iddfile)
    # import pdb; pdb.set_trace()
    block, data, commdct, idd_index = readidf.readdatacommdct1(
        fname,
        iddfile=iddfile,
        commdct=commdct,
        block=block)
    if conv:
        convertallfields(data, commdct, block)
    # fill gaps in idd
    ddtt, dtls = data.dt, data.dtls
    if versiontuple < (8,):
        skiplist = ["TABLE:MULTIVARIABLELOOKUP"]
    else:
        skiplist = None
    nofirstfields = iddgaps.missingkeys_standard(
        commdct, dtls,
        skiplist=skiplist)
    iddgaps.missingkeys_nonstandard(block, commdct, dtls, nofirstfields)
    # bunchdt = makebunches(data, commdct)
    bunchdt = makebunches_alter(data, commdct, theidf)
    return bunchdt, block, data, commdct, idd_index, versiontuple