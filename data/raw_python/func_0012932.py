def getnodefieldname(idfobject, endswith, fluid=None, startswith=None):
    """return the field name of the node
    fluid is only needed if there are air and water nodes
    fluid is Air or Water or ''.
    if the fluid is Steam, use Water"""
    if startswith is None:
        startswith = ''
    if fluid is None:
        fluid = ''
    nodenames = getfieldnamesendswith(idfobject, endswith)
    nodenames = [name for name in nodenames if name.startswith(startswith)]
    fnodenames = [nd for nd in nodenames if nd.find(fluid) != -1]
    fnodenames = [name for name in fnodenames if name.startswith(startswith)]
    if len(fnodenames) == 0:
        nodename = nodenames[0]
    else:
        nodename = fnodenames[0]
    return nodename