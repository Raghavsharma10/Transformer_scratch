def readdatacommlst(idfname):
    """read the idf file"""
    # iddfile = sys.path[0] + '/EplusCode/Energy+.idd'
    iddfile = 'Energy+.idd'
    # iddfile = './EPlusInterfaceFunctions/E1.idd' # TODO : can the path name be not hard coded
    iddtxt = open(iddfile, 'r').read()
    block, commlst, commdct = parse_idd.extractidddata(iddfile)

    theidd = eplusdata.Idd(block, 2)
    data = eplusdata.Eplusdata(theidd, idfname)
    return data, commlst