def getPart(ID, chart):
    """ Returns an Arabic Part. """
    obj = GenericObject()
    obj.id = ID
    obj.type = const.OBJ_ARABIC_PART
    obj.relocate(partLon(ID, chart))
    return obj