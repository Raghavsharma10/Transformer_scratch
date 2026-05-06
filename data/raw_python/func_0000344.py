def changeset(python_data: LdapObject, d: dict) -> Changeset:
    """ Generate changes object for ldap object. """
    table: LdapObjectClass = type(python_data)
    fields = table.get_fields()
    changes = Changeset(fields, src=python_data, d=d)
    return changes