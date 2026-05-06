def getobjectswithnode(idf, nodekeys, nodename):
    """return all objects that mention this node name"""
    keys = nodekeys
    # TODO getidfkeyswithnodes needs to be done only once. take out of here
    listofidfobjects = (idf.idfobjects[key.upper()] 
                for key in keys if idf.idfobjects[key.upper()])
    idfobjects = [idfobj 
                    for idfobjs in listofidfobjects 
                        for idfobj in idfobjs]
    objwithnodes = []
    for obj in idfobjects:
        values = obj.fieldvalues
        fdnames = obj.fieldnames
        for value, fdname in zip(values, fdnames):
            if fdname.endswith('Node_Name'):
                if value == nodename:
                    objwithnodes.append(obj)
                    break
    return objwithnodes