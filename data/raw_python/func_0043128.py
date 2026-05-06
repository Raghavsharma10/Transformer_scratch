def DataLookup(fieldVal, db, lookupType, fieldName, histObj={}):
    """
    Return new field value based on single-value lookup against MongoDB

    :param string fieldVal: input value to lookup
    :param MongoClient db: MongoClient instance connected to MongoDB
    :param string lookupType: Type of lookup to perform/MongoDB collection name.
           One of 'genericLookup', 'fieldSpecificLookup', 'normLookup'
    :param string fieldName: Field name to query against
    :param dict histObj: History object to which changes should be appended
    """

    if lookupType == 'genericLookup':
        lookup_dict = {"find": _DataClean_(fieldVal)}
    elif lookupType in ['fieldSpecificLookup', 'normLookup']:
        lookup_dict = {"fieldName": fieldName, "find": _DataClean_(fieldVal)}
    else:
        raise ValueError("Invalid lookupType")

    field_val_new = fieldVal

    coll = db[lookupType]

    l_val = coll.find_one(lookup_dict, ['replace'])

    if l_val:
        field_val_new = l_val['replace'] if 'replace' in l_val else ''

    change = _CollectHistory_(lookupType=lookupType, fromVal=fieldVal,
                              toVal=field_val_new)

    hist_obj_upd = _CollectHistoryAgg_(contactHist=histObj,
                                       fieldHistObj=change,
                                       fieldName=fieldName)

    return field_val_new, hist_obj_upd