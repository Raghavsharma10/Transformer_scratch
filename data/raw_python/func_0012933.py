def connectcomponents(idf, components, fluid=None):
    """rename nodes so that the components get connected
    fluid is only needed if there are air and water nodes
    fluid is Air or Water or ''.
    if the fluid is Steam, use Water"""
    if fluid is None:
        fluid = ''
    if len(components) == 1:
        thiscomp, thiscompnode = components[0]
        initinletoutlet(idf, thiscomp, thiscompnode, force=False)
        outletnodename = getnodefieldname(thiscomp, "Outlet_Node_Name",
                                          fluid=fluid, startswith=thiscompnode)
        thiscomp[outletnodename] = [thiscomp[outletnodename],
                                    thiscomp[outletnodename]]
        # inletnodename = getnodefieldname(nextcomp, "Inlet_Node_Name", fluid)
        # nextcomp[inletnodename] = [nextcomp[inletnodename], betweennodename]
        return components
    for i in range(len(components) - 1):
        thiscomp, thiscompnode = components[i]
        nextcomp, nextcompnode = components[i + 1]
        initinletoutlet(idf, thiscomp, thiscompnode, force=False)
        initinletoutlet(idf, nextcomp, nextcompnode, force=False)
        betweennodename = "%s_%s_node" % (thiscomp.Name, nextcomp.Name)
        outletnodename = getnodefieldname(thiscomp, "Outlet_Node_Name",
                                          fluid=fluid, startswith=thiscompnode)
        thiscomp[outletnodename] = [thiscomp[outletnodename], betweennodename]
        inletnodename = getnodefieldname(nextcomp, "Inlet_Node_Name", fluid)
        nextcomp[inletnodename] = [nextcomp[inletnodename], betweennodename]
    return components