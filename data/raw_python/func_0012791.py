def tabstr2list(data):
    """tabstr2list"""
    alist = data.split(os.linesep)
    blist = alist[1].split('\t')

    clist = []
    for num in range(0, len(alist)):
        ilist = alist[num].split('\t')
        clist = clist+[ilist]
    cclist = clist[:-1]
      #the last element is turning out to be empty
      #this is because the string ends with a os.linesep
    return cclist