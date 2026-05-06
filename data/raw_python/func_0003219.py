def create_dhcp_options(input_dict, ignoreError = False, generateNone = False):
    """
    Try best to create dhcp_options from human friendly values, ignoring
    invalid values
    """
    retdict = {}
    for k,v in dict(input_dict).items():
        try:
            if generateNone and v is None:
                retdict[k] = None
            else:
                try:
                    retdict[k] = create_option_from_value(k, v)
                except _EmptyOptionException:
                    if generateNone:
                        retdict[k] = None
        except Exception:
            if ignoreError:
                continue
            else:
                raise
    return retdict