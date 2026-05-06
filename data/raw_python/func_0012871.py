def get_nocom_vars(astr):
    """
    input 'astr' which is the Energy+.idd file as a string
    returns (st1, st2, lss)
    st1 = with all the ! comments striped
    st2 = strips all comments - both the '!' and '\\'
    lss = nested list of all the variables in Energy+.idd file
    """
    nocom = nocomment(astr, '!')# remove '!' comments
    st1 = nocom
    nocom1 = nocomment(st1, '\\')# remove '\' comments
    st1 = nocom
    st2 = nocom1
    # alist = string.split(st2, ';')
    alist = st2.split(';')
    lss = []

    # break the .idd file into a nested list
    #=======================================
    for element in alist:
        # item = string.split(element, ',')
        item = element.split(',')
        lss.append(item)
    for i in range(0, len(lss)):
        for j in range(0, len(lss[i])):
            lss[i][j] = lss[i][j].strip()
    if len(lss) > 1:
        lss.pop(-1)
    #=======================================

    #st1 has the '\' comments --- looks like I don't use this
    #lss is the .idd file as a nested list
    return (st1, st2, lss)