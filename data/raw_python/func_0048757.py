def _toDict(datastorage_obj,recursive=True):
    """ this is the recursive part of the toDict (otherwise it fails when converting to DataStorage """
    if "items" not in dir(datastorage_obj): return datastorage_obj
    d = dict()
    for k, v in datastorage_obj.items():
        try:
            d[k] = _toDict(v)
        except Exception as e:
            log.info("In toDict, could not convert key %s to dict, error was %s" %
                     (k, e))
            d[k] = v
    return d