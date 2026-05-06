def fix(source_key, rownum, line, source, sample_only_filter, sample_size):
    """
    :param rownum:
    :param line:
    :param source:
    :param sample_only_filter:
    :param sample_size:
    :return:  (row, no_more_data) TUPLE WHERE row IS {"value":<data structure>} OR {"json":<text line>}
    """
    value = json2value(line)

    if rownum == 0:
        if len(line) > MAX_RECORD_LENGTH:
            _shorten(source_key, value, source)
        value = _fix(value)
        if sample_only_filter and Random.int(int(1.0/coalesce(sample_size, 0.01))) != 0 and jx.filter([value], sample_only_filter):
            # INDEX etl.id==0, BUT NO MORE
            if value.etl.id != 0:
                Log.error("Expecting etl.id==0")
            row = {"value": value}
            return row, True
    elif len(line) > MAX_RECORD_LENGTH:
        _shorten(source_key, value, source)
        value = _fix(value)
    elif '"resource_usage":' in line:
        value = _fix(value)

    row = {"value": value}
    return row, False