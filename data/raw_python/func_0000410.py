def str2dn(dn, flags=0):
    """
    This function takes a DN as string as parameter and returns
    a decomposed DN. It's the inverse to dn2str().

    flags describes the format of the dn

    See also the OpenLDAP man-page ldap_str2dn(3)
    """

    # if python2, we need unicode string
    if not isinstance(dn, six.text_type):
        dn = dn.decode("utf_8")

    assert flags == 0
    result, i = _distinguishedName(dn, 0)
    if result is None:
        raise tldap.exceptions.InvalidDN("Cannot parse dn")
    if i != len(dn):
        raise tldap.exceptions.InvalidDN("Cannot parse dn past %s" % dn[i:])
    return result