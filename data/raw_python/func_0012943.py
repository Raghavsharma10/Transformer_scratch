def main():
    """the main routine"""
    from six import StringIO
    import eppy.iddv7 as iddv7
    IDF.setiddname(StringIO(iddv7.iddtxt))
    idf1 = IDF(StringIO(''))
    loopname = "p_loop"
    sloop = ['sb0', ['sb1', 'sb2', 'sb3'], 'sb4']
    dloop = ['db0', ['db1', 'db2', 'db3'], 'db4']
    # makeplantloop(idf1, loopname, sloop, dloop)
    loopname = "c_loop"
    sloop = ['sb0', ['sb1', 'sb2', 'sb3'], 'sb4']
    dloop = ['db0', ['db1', 'db2', 'db3'], 'db4']
    # makecondenserloop(idf1, loopname, sloop, dloop)
    loopname = "a_loop"
    sloop = ['sb0', ['sb1', 'sb2', 'sb3'], 'sb4']
    dloop = ['zone1', 'zone2', 'zone3']
    makeairloop(idf1, loopname, sloop, dloop)
    idf1.savecopy("hh1.idf")