def getbranchcomponents(idf, branch, utest=False):
    """get the components of the branch"""
    fobjtype = 'Component_%s_Object_Type'
    fobjname = 'Component_%s_Name'
    complist = []
    for i in range(1, 100000):
        try:
            objtype = branch[fobjtype % (i,)]
            if objtype.strip() == '':
                break
            objname = branch[fobjname % (i,)]
            complist.append((objtype, objname))
        except bunch_subclass.BadEPFieldError:
            break
    if utest:
        return complist
    else:
        return [idf.getobject(ot, on) for ot, on in complist]