def tabfile2doefile(tabfile, doefile):
    """tabfile2doefile"""
    alist = tabfile2list(tabfile)
    astr = list2doe(alist)
    mylib1.write_str2file(doefile, astr)