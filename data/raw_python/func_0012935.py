def componentsintobranch(idf, branch, listofcomponents, fluid=None):
    """insert a list of components into a branch
    fluid is only needed if there are air and water nodes in same object
    fluid is Air or Water or ''.
    if the fluid is Steam, use Water"""
    if fluid is None:
        fluid = ''
    componentlist = [item[0] for item in listofcomponents]
    # assumes that the nodes of the component connect to each other
    # empty branch if it has existing components
    thebranchname = branch.Name
    thebranch = idf.removeextensibles('BRANCH', thebranchname)  # empty the branch
    # fill in the new components with the node names into this branch
        # find the first extensible field and fill in the data in obj.
    e_index = idf.getextensibleindex('BRANCH', thebranchname)
    theobj = thebranch.obj
    modeleditor.extendlist(theobj, e_index)  # just being careful here
    for comp, compnode in listofcomponents:
        theobj.append(comp.key)
        theobj.append(comp.Name)
        inletnodename = getnodefieldname(comp, "Inlet_Node_Name", fluid=fluid,
                                         startswith=compnode)
        theobj.append(comp[inletnodename])
        outletnodename = getnodefieldname(comp, "Outlet_Node_Name",
                                          fluid=fluid, startswith=compnode)
        theobj.append(comp[outletnodename])
        theobj.append('')

    return thebranch