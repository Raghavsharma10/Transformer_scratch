def clean_item_no_list(i):
    """
        Return a json-clean item or None. Will log info message for failure.
    """
    itype = type(i)
    if itype == dict:
        return clean_dict(i, clean_item_no_list)
    elif itype == list:
        return clean_tuple(i, clean_item_no_list)
    elif itype == tuple:
        return clean_tuple(i, clean_item_no_list)
    elif itype == numpy.float32:
        return float(i)
    elif itype == numpy.float64:
        return float(i)
    elif itype == numpy.int16:
        return int(i)
    elif itype == numpy.uint16:
        return int(i)
    elif itype == numpy.int32:
        return int(i)
    elif itype == numpy.uint32:
        return int(i)
    elif itype == float:
        return i
    elif itype == str:
        return i
    elif itype == int:
        return i
    elif itype == bool:
        return i
    elif itype == type(None):
        return i
    logging.info("[2] Unable to handle type %s", itype)
    return None