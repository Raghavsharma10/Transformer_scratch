def _doPrep(field_dict):
    """
    _doPrep is makes changes in-place.
    Do some prep work converting python types into formats that
    Salesforce will accept.
    This includes converting lists of strings to "apple;orange;pear".
    Dicts will be converted to embedded objects
    None or empty list values will be Null-ed
    """
    fieldsToNull = []
    for key, value in field_dict.items():
        if value is None:
            fieldsToNull.append(key)
            field_dict[key] = []
        if hasattr(value, '__iter__'):
            if len(value) == 0:
                fieldsToNull.append(key)
            elif isinstance(value, dict):
                innerCopy = copy.deepcopy(value)
                _doPrep(innerCopy)
                field_dict[key] = innerCopy
            else:
                field_dict[key] = ";".join(value)
    if 'fieldsToNull' in field_dict:
        raise ValueError(
            "fieldsToNull should be populated by the client, not the caller."
        )
    field_dict['fieldsToNull'] = fieldsToNull