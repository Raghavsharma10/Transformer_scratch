def replacebranch1(idf, loop, branchname, listofcomponents_tuples, fluid=None,
                   debugsave=False):
    """do I even use this ? .... yup! I do"""
    if fluid is None:
        fluid = ''
    listofcomponents_tuples = _clean_listofcomponents_tuples(listofcomponents_tuples)
    branch = idf.getobject('BRANCH', branchname)  # args are (key, name)
    listofcomponents = []
    for comp_type, comp_name, compnode in listofcomponents_tuples:
        comp = getmakeidfobject(idf, comp_type.upper(), comp_name)
        listofcomponents.append((comp, compnode))
    newbr = replacebranch(idf, loop, branch, listofcomponents,
                          debugsave=debugsave, fluid=fluid)
    return newbr