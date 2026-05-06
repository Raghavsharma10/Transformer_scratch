def is_not_null_predicate(
    raw_crash, dumps, processed_crash, processor, key=''
):
    """a predicate that converts the key'd source to boolean.

    parameters:
        raw_crash - dict
        dumps - placeholder in a fat interface - unused
        processed_crash - placeholder in a fat interface - unused
        processor - placeholder in a fat interface - unused
    """
    try:
        return bool(raw_crash[key])
    except KeyError:
        return False