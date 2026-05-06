def tabfile2list(fname):
    "tabfile2list"
    #dat = mylib1.readfileasmac(fname)
    #data = string.strip(dat)
    data = mylib1.readfileasmac(fname)
    #data = data[:-2]#remove the last return
    alist = data.split('\r')#since I read it as a mac file
    blist = alist[1].split('\t')

    clist = []
    for num in range(0, len(alist)):
        ilist = alist[num].split('\t')
        clist = clist+[ilist]
    cclist = clist[:-1]#the last element is turning out to be empty
    return cclist