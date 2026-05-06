def get_censor_char():
    """Plucks a letter out of the censor_pool. If the censor_pool is empty,
    replenishes it. This is done to ensure all censor chars are used before
    grabbing more (avoids ugly duplicates).

    """
    global _censor_pool
    if not _censor_pool:
        # censor pool is empty. fill it back up.
        _censor_pool = list(_censor_chars)
    return _censor_pool.pop(random.randrange(len(_censor_pool)))