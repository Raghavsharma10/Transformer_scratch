def getfields(comm):
    """get all the fields that have the key 'field' """
    fields = []
    for field in comm:
        if 'field' in field:
            fields.append(field)
    return fields