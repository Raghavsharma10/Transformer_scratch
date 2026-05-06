def explode_dn(dn, notypes=0, flags=0):
    """
    explode_dn(dn [, notypes=0]) -> list

    This function takes a DN and breaks it up into its component parts.
    The notypes parameter is used to specify that only the component's
    attribute values be returned and not the attribute types.
    """
    if not dn:
        return []
    dn_decomp = str2dn(dn, flags)
    rdn_list = []
    for rdn in dn_decomp:
        if notypes:
            rdn_list.append('+'.join([
                escape_dn_chars(avalue or '')
                for atype, avalue, dummy in rdn
            ]))
        else:
            rdn_list.append('+'.join([
                '='.join((atype, escape_dn_chars(avalue or '')))
                for atype, avalue, dummy in rdn
            ]))
    return rdn_list