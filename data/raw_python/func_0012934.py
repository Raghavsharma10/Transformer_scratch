def initinletoutlet(idf, idfobject, thisnode, force=False):
    """initialze values for all the inlet outlet nodes for the object.
    if force == False, it willl init only if field = '' """
    def blankfield(fieldvalue):
        """test for blank field"""
        try:
            if fieldvalue.strip() == '':
                return True
            else:
                return False
        except AttributeError:  # field may be a list
            return False
    def trimfields(fields, thisnode):
        if len(fields) > 1:
            if thisnode is not None:
                fields = [field for field in fields
                          if field.startswith(thisnode)]
                return fields
            else:
                print("Where should this loop connect ?")
                print("%s - %s" % (idfobject.key, idfobject.Name))
                print([field.split("Inlet_Node_Name")[0]
                       for field in inletfields])
                raise WhichLoopError
        else:
            return fields

    inletfields = getfieldnamesendswith(idfobject, "Inlet_Node_Name")
    inletfields = trimfields(inletfields, thisnode)  # or warn with exception
    for inletfield in inletfields:
        if blankfield(idfobject[inletfield]) == True or force == True:
            idfobject[inletfield] = "%s_%s" % (idfobject.Name, inletfield)
    outletfields = getfieldnamesendswith(idfobject, "Outlet_Node_Name")
    outletfields = trimfields(outletfields, thisnode)  # or warn with exception
    for outletfield in outletfields:
        if blankfield(idfobject[outletfield]) == True or force == True:
            idfobject[outletfield] = "%s_%s" % (idfobject.Name, outletfield)
    return idfobject