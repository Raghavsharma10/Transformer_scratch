def json_encode_default(obj):
    '''
    Convert datetime.datetime to timestamp

    :param obj: value to (possibly) convert
    '''
    if isinstance(obj, (datetime, date)):
        result = dt2ts(obj)
    else:
        result = json_encoder.default(obj)
    return to_encoding(result)