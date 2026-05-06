def IncludesLookup(fieldVal, lookupType, db, fieldName, deriveFieldName='',
                   deriveInput={}, histObj={}, overwrite=False,
                   blankIfNoMatch=False):
    """
    Return new field value based on whether or not original value includes AND
    excludes all words in a comma-delimited list queried from MongoDB

    :param string fieldVal: input value to lookup
    :param string lookupType: Type of lookup to perform/MongoDB collection name.
           One of 'normIncludes', 'deriveIncludes'
    :param MongoClient db: MongoClient instance connected to MongoDB
    :param string fieldName: Field name to query against
    :param string deriveFieldName: Field name from which to derive value
    :param dict deriveInput: Values to perform lookup against:
           {"deriveFieldName": "deriveVal1"}
    :param dict histObj: History object to which changes should be appended
    :param bool overwrite: Should an existing field value be replaced
    :param bool blankIfNoMatch: Should field value be set to blank if
           no match is found
    """

    lookup_dict = {
        'fieldName': fieldName
    }

    if lookupType == 'normIncludes':
        field_val_clean = _DataClean_(fieldVal)

    elif lookupType == 'deriveIncludes':

        if deriveFieldName == '' or deriveInput == {}:
            raise ValueError("for 'deriveIncludes' must specify both \
                              'deriveFieldName' and 'deriveInput'")

        lookup_dict['deriveFieldName'] = deriveFieldName
        field_val_clean = _DataClean_(deriveInput[list(deriveInput.keys())[0]])
    else:
        raise ValueError("Invalid lookupType")

    field_val_new = fieldVal
    check_match = False
    using = {}

    coll = db[lookupType]

    inc_val = coll.find(lookup_dict, ['includes', 'excludes', 'begins', 'ends',
                                      'replace'])

    if inc_val and (lookupType == 'normIncludes' or
                    (lookupType == 'deriveIncludes' and
                     (overwrite or fieldVal == ''))):

        for row in inc_val:

            try:

                if (row['includes'] != '' or
                        row['excludes'] != '' or
                        row['begins'] != '' or
                        row['ends'] != ''):

                    if all((a in field_val_clean)
                           for a in row['includes'].split(",")):

                        if all((b not in field_val_clean)
                               for b in row['excludes'].split(",")) \
                                or row['excludes'] == '':

                            if field_val_clean.startswith(row['begins']):

                                if field_val_clean.endswith(row['ends']):

                                    field_val_new = row['replace']

                                    if lookupType == 'deriveIncludes':
                                        using[deriveFieldName] = deriveInput

                                    using['includes'] = row['includes']
                                    using['excludes'] = row['excludes']
                                    using['begins'] = row['begins']
                                    using['ends'] = row['ends']

                                    check_match = True

                                    break

            except KeyError as Key_error_obj:
                warnings.warn('schema error', Key_error_obj)

        if inc_val:
            inc_val.close()

    if (field_val_new == fieldVal and blankIfNoMatch and
            lookupType == 'deriveIncludes'):
        field_val_new = ''
        using['blankIfNoMatch'] = 'no match found'

    change = _CollectHistory_(lookupType=lookupType, fromVal=fieldVal,
                              toVal=field_val_new, using=using)

    histObjUpd = _CollectHistoryAgg_(contactHist=histObj, fieldHistObj=change,
                                     fieldName=fieldName)

    return field_val_new, histObjUpd, check_match