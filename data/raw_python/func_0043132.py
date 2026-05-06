def DeriveDataCopyValue(fieldName, deriveInput, overwrite, fieldVal, histObj={}):
    """
    Return new value based on value from another field

    :param string fieldName: Field name to query against
    :param dict deriveInput: Values to perform lookup against:
           {"copyField1": "copyVal1"}
    :param bool overwrite: Should an existing field value be replaced
    :param string fieldVal: Current field value
    :param dict histObj: History object to which changes should be appended
    """

    if len(deriveInput) > 1:
        raise Exception("more than one field/value in deriveInput")

    field_val_new = fieldVal

    row = list(deriveInput.keys())[0]

    if deriveInput[row] != '' and (overwrite or (fieldVal == '')):
        field_val_new = deriveInput[row]
        check_match = True
    else:
        check_match = False

    change = _CollectHistory_(lookupType='copyValue', fromVal=fieldVal,
                              toVal=field_val_new, using=deriveInput)

    hist_obj_upd = _CollectHistoryAgg_(contactHist=histObj, fieldHistObj=change,
                                       fieldName=fieldName)

    return field_val_new, hist_obj_upd, check_match