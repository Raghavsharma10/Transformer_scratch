def DeriveDataRegex(fieldName, db, deriveInput, overwrite, fieldVal, histObj={},
                    blankIfNoMatch=False):
    """
    Return a new field value based on match (of another field) against regex
    queried from MongoDB

    :param string fieldName: Field name to query against
    :param MongoClient db: MongoClient instance connected to MongoDB
    :param dict deriveInput: Values to perform lookup against:
           {"lookupField1": "lookupVal1"}
    :param bool overwrite: Should an existing field value be replaced
    :param string fieldVal: Current field value
    :param dict histObj: History object to which changes should be appended
    :param bool blankIfNoMatch: Should field value be set to blank
           if no match is found
    """

    if len(deriveInput) > 1:
        raise Exception("more than one value in deriveInput")

    field_val_new = fieldVal
    check_match = False

    # derive_using = deriveInput

    row = list(deriveInput.keys())[0]

    pattern = ''

    if deriveInput[row] != '' and (overwrite or (fieldVal == '')):

        lookup_dict = {
            'deriveFieldName': row,
            'fieldName': fieldName
        }

        coll = db['deriveRegex']

        re_val = coll.find(lookup_dict, ['pattern', 'replace'])

        for l_val in re_val:

            try:

                match = re.match(l_val['pattern'],
                                 _DataClean_(deriveInput[row]),
                                 flags=re.IGNORECASE)

                if match:

                    field_val_new = re.sub(l_val['pattern'], l_val['replace'],
                                           _DataClean_(deriveInput[row]),
                                           flags=re.IGNORECASE)

                    pattern = l_val['pattern']

                    check_match = True
                    break

            except KeyError as key_error_obj:
                warnings.warn('schema error', key_error_obj)

        if re_val:
            re_val.close()

        if field_val_new == fieldVal and blankIfNoMatch:
            field_val_new = ''
            pattern = 'no matching pattern'
            # derive_using = {"blankIfNoMatch": "no match found"}

    change = _CollectHistory_(lookupType='deriveRegex', fromVal=fieldVal,
                              toVal=field_val_new, using=deriveInput,
                              pattern=pattern)

    hist_obj_upd = _CollectHistoryAgg_(contactHist=histObj, fieldHistObj=change,
                                       fieldName=fieldName)

    return field_val_new, hist_obj_upd, check_match