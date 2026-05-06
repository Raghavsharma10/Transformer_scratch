def valid_hash_value(hashname):
    """Return true if given value is a valid, recommended hash name according
    to the STIX 2 specification.
    """
    custom_hash_prefix_re = re.compile(r"^x_")
    if hashname in enums.HASH_ALGO_OV or custom_hash_prefix_re.match(hashname):
        return True
    else:
        return False