def _CollectHistoryAgg_(contactHist, fieldHistObj, fieldName):
    """
    Return updated history dictionary with new field change

    :param dict contactHist: Existing contact history dictionary
    :param dict fieldHistObj: Output of _CollectHistory_
    :param string fieldName: field name
    """

    if fieldHistObj!={}:
        if fieldName not in contactHist.keys():
            contactHist[fieldName] = {}
        for lookupType in fieldHistObj.keys():
            contactHist[fieldName][lookupType] = fieldHistObj[lookupType]

    return contactHist