def rdn_to_dn(changes: Changeset, name: str, base_dn: str) -> Changeset:
    """ Convert the rdn to a fully qualified DN for the specified LDAP
    connection.

    :param changes: The changes object to lookup.
    :param name: rdn to convert.
    :param base_dn: The base_dn to lookup.
    :return: fully qualified DN.
    """
    dn = changes.get_value_as_single('dn')
    if dn is not None:
        return changes

    value = changes.get_value_as_single(name)
    if value is None:
        raise tldap.exceptions.ValidationError(
            "Cannot use %s in dn as it is None" % name)
    if isinstance(value, list):
        raise tldap.exceptions.ValidationError(
            "Cannot use %s in dn as it is a list" % name)

    assert base_dn is not None

    split_base = str2dn(base_dn)
    split_new_dn = [[(name, value, 1)]] + split_base

    new_dn = dn2str(split_new_dn)

    return changes.set('dn', new_dn)