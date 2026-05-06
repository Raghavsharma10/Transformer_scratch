def _get_field_by_name(table: LdapObjectClass, name: str) -> tldap.fields.Field:
    """ Lookup a field by its name. """
    fields = table.get_fields()
    return fields[name]