def get_referenced_object(referring_object, fieldname):
    """
    Get an object referred to by a field in another object.

    For example an object of type Construction has fields for each layer, each
    of which refers to a Material. This functions allows the object
    representing a Material to be fetched using the name of the layer.

    Returns the first item found since if there is more than one matching item,
    it is a malformed IDF.

    Parameters
    ----------
    referring_object : EpBunch
        The object which contains a reference to another object,
    fieldname : str
        The name of the field in the referring object which contains the
        reference to another object.

    Returns
    -------
    EpBunch

    """
    idf = referring_object.theidf
    object_list = referring_object.getfieldidd_item(fieldname, u'object-list')
    for obj_type in idf.idfobjects:
        for obj in idf.idfobjects[obj_type]:
            valid_object_lists = obj.getfieldidd_item("Name", u'reference')
            if set(object_list).intersection(set(valid_object_lists)):
                referenced_obj_name = referring_object[fieldname]
                if obj.Name == referenced_obj_name:
                    return obj