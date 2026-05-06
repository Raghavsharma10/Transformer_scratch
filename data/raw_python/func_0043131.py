def DeriveDataLookup(fieldName, db, deriveInput, overwrite=True, fieldVal='',
                     histObj={}, blankIfNoMatch=False):
    """
    Return new field value based on single or multi-value lookup against MongoDB

    :param string fieldName: Field name to query against
    :param MongoClient db: MongoClient instance connected to MongoDB
    :param dict deriveInput: Values to perform lookup against:
           {"lookupField1": "lookupVal1", "lookupField2": "lookupVal2"}
    :param bool overwrite: Should an existing field value be replaced
    :param string fieldVal: Current field value
    :param dict histObj: History object to which changes should be appended
    :param bool blankIfNoMatch: Should field value be set to blank
           if no match is found
    """

    lookup_vals = OrderedDict()

    for val in sorted(deriveInput.keys()):
        lookup_vals[val] = _DataClean_(deriveInput[val])

    lookup_dict = {
        'fieldName': fieldName,
        'lookupVals': lookup_vals
    }

    coll = db['deriveValue']

    l_val = coll.find_one(lookup_dict, ['value'])

    field_val_new = fieldVal

    derive_using = deriveInput

    # If match found return True else False
    check_match = True if l_val else False

    if l_val and (overwrite or (fieldVal == '')):

        try:
            field_val_new = l_val['value']
        except KeyError as Key_error_obj:
            warnings.warn('schema error', Key_error_obj)

    elif blankIfNoMatch and not l_val:

        field_val_new = ''
        derive_using = {'blankIfNoMatch': 'no match found'}

    change = _CollectHistory_(lookupType='deriveValue', fromVal=fieldVal,
                              toVal=field_val_new, using=derive_using)

    hist_obj_upd = _CollectHistoryAgg_(contactHist=histObj, fieldHistObj=change,
                                       fieldName=fieldName)

    return field_val_new, hist_obj_upd, check_match