def RegexLookup(fieldVal, db, fieldName, lookupType, histObj={}):
    """
    Return a new field value based on match against regex queried from MongoDB

    :param string fieldVal: input value to lookup
    :param MongoClient db: MongoClient instance connected to MongoDB
    :param string lookupType: Type of lookup to perform/MongoDB collection name.
            One of 'genericRegex', 'fieldSpecificRegex', 'normRegex'
    :param string fieldName: Field name to query against
    :param dict histObj: History object to which changes should be appended
    """

    if lookupType == 'genericRegex':
        lookup_dict = {}
    elif lookupType in ['fieldSpecificRegex', 'normRegex']:
        lookup_dict = {"fieldName": fieldName}
    else:
        raise ValueError("Invalid type")

    field_val_new = fieldVal
    pattern = ''

    coll = db[lookupType]

    re_val = coll.find(lookup_dict, ['pattern', 'replace'])

    for row in re_val:

        try:
            match = re.match(row['pattern'], _DataClean_(field_val_new),
                             flags=re.IGNORECASE)

            if match:

                if 'replace' in row:
                    field_val_new = re.sub(row['pattern'], row['replace'],
                                           _DataClean_(field_val_new),
                                           flags=re.IGNORECASE)
                else:
                    field_val_new = re.sub(row['pattern'], '',
                                           _DataClean_(field_val_new),
                                           flags=re.IGNORECASE)

                pattern = row['pattern']
                break

        except KeyError as Key_error_obj:
            warnings.warn('schema error', Key_error_obj)

    if re_val:
        re_val.close()

    change = _CollectHistory_(lookupType=lookupType, fromVal=fieldVal,
                              toVal=field_val_new, pattern=pattern)

    histObjUpd = _CollectHistoryAgg_(contactHist=histObj, fieldHistObj=change,
                                     fieldName=fieldName)

    return field_val_new, histObjUpd