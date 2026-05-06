def mtabstr2doestr(st1):
    """mtabstr2doestr"""
    seperator = '$ =============='
    alist = st1.split(seperator)

    #this removes all the tabs that excel
    #puts after the seperator and before the next line
    for num in range(0, len(alist)):
        alist[num] = alist[num].lstrip()
    st2 = ''
    for num in range(0, len(alist)):
        alist = tabstr2list(alist[num])
        st2 = st2 + list2doe(alist)

    lss = st2.split('..')
    mylib1.write_str2file('forfinal.txt', st2)#for debugging
    print(len(lss))


    st3 = tree2doe(st2)
    lsss = st3.split('..')
    print(len(lsss))
    return st3