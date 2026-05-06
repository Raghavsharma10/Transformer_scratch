def explode_rdn(rdn, notypes=0, flags=0):
    """
    explode_rdn(rdn [, notypes=0]) -> list

    This function takes a RDN and breaks it up into its component parts
    if it is a multi-valued RDN.
    The notypes parameter is used to specify that only the component's
    attribute values be returned and not the attribute types.
    """
    if not rdn:
        return []
    rdn_decomp = str2dn(rdn, flags)[0]
    if notypes:
        return [avalue or '' for atype, avalue, dummy in rdn_decomp]
    else:
        return ['='.join((atype, escape_dn_chars(avalue or '')))
                for atype, avalue, dummy in rdn_decomp]