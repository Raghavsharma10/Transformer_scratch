def doestr2tabstr(astr, kword):
    """doestr2tabstr"""
    alist = astr.split('..')
    del astr
    #strip junk put .. back
    for num in range(0, len(alist)):
        alist[num] = alist[num].strip()
        alist[num] = alist[num] + os.linesep + '..' + os.linesep
    alist.pop()

    lblock = []
    for num in range(0, len(alist)):
        linels = alist[num].split(os.linesep)
        firstline = linels[0]
        assignls = firstline.split('=')
        keyword = assignls[-1].strip()
        if keyword == kword:
            lblock = lblock + [alist[num]]
            #print firstline

    #get all val
    lval = []
    for num in range(0, len(lblock)):
        block = lblock[num]
        linel = block.split(os.linesep)
        lvalin = []
        for k in range(0, len(linel)):
            line = linel[k]
            assignl = line.split('=')
            if k == 0:
                lvalin = lvalin + [assignl[0]]
            else:
                if assignl[-1] == '..':
                    assignl[-1] = '.'
                lvalin = lvalin + [assignl[-1]]
        lvalin.pop()
        lval = lval + [lvalin]

    #get keywords
    kwordl = []
    block = lblock[0]
    linel = block.split(os.linesep)
    for k in range(0, len(linel)):
        line = linel[k]
        assignl = line.split('=')
        if k == 0:
            kword = ' =  ' + assignl[1].strip()
        else:
            if assignl[0] == '..':
                assignl[0] = '.'
            else:
                assignl[0] = assignl[0] + '='
            kword = assignl[0].strip()
        kwordl = kwordl + [kword]
    kwordl.pop()

    astr = ''
    for num in range(0, len(kwordl)):
        linest = ''
        linest = linest + kwordl[num]
        for k in range(0, len(lval)):
            linest = linest + '\t' + lval[k][num]
        astr = astr + linest + os.linesep

    return astr