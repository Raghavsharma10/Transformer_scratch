def _python_to_mod_modify(changes: Changeset) -> Dict[str, List[Tuple[Operation, List[bytes]]]]:
    """ Convert a LdapChanges object to a modlist for a modify operation. """
    table: LdapObjectClass = type(changes.src)
    changes = changes.changes

    result: Dict[str, List[Tuple[Operation, List[bytes]]]] = {}
    for key, l in changes.items():
        field = _get_field_by_name(table, key)

        if field.db_field:
            try:
                new_list = [
                    (operation, field.to_db(value))
                    for operation, value in l
                ]
                result[key] = new_list
            except ValidationError as e:
                raise ValidationError(f"{key}: {e}.")

    return result