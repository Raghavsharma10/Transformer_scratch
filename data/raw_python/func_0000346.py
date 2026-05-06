def _python_to_mod_new(changes: Changeset) -> Dict[str, List[List[bytes]]]:
    """ Convert a LdapChanges object to a modlist for add operation. """
    table: LdapObjectClass = type(changes.src)
    fields = table.get_fields()

    result: Dict[str, List[List[bytes]]] = {}

    for name, field in fields.items():
        if field.db_field:
            try:
                value = field.to_db(changes.get_value_as_list(name))
                if len(value) > 0:
                    result[name] = value
            except ValidationError as e:
                raise ValidationError(f"{name}: {e}.")

    return result