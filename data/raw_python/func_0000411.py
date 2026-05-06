def dn2str(dn):
    """
    This function takes a decomposed DN as parameter and returns
    a single string. It's the inverse to str2dn() but will always
    return a DN in LDAPv3 format compliant to RFC 4514.
    """
    for rdn in dn:
        for atype, avalue, dummy in rdn:
            assert isinstance(atype, six.string_types)
            assert isinstance(avalue, six.string_types)
            assert dummy == 1

    return ','.join([
        '+'.join([
            '='.join((atype, escape_dn_chars(avalue or '')))
            for atype, avalue, dummy in rdn])
        for rdn in dn
    ])