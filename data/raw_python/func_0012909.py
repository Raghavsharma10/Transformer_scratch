def getobject_use_prevfield(idf, idfobject, fieldname):
    """field=object_name, prev_field=object_type. Return the object"""
    if not fieldname.endswith("Name"):
        return None
    # test if prevfieldname ends with "Object_Type"
    fdnames = idfobject.fieldnames
    ifieldname = fdnames.index(fieldname)
    prevfdname = fdnames[ifieldname - 1]
    if not prevfdname.endswith("Object_Type"):
        return None
    objkey = idfobject[prevfdname].upper()
    objname = idfobject[fieldname]
    try:
        foundobj = idf.getobject(objkey, objname)
    except KeyError as e:
        return None
    return foundobj