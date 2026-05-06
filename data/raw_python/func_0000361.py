def modifyModlist(
        old_entry: dict, new_entry: dict, ignore_attr_types: Optional[List[str]] = None,
        ignore_oldexistent: bool = False) -> Dict[str, Tuple[str, List[bytes]]]:
    """
    Build differential modify list for calling LDAPObject.modify()/modify_s()

    :param old_entry:
        Dictionary holding the old entry
    :param new_entry:
        Dictionary holding what the new entry should be
    :param ignore_attr_types:
        List of attribute type names to be ignored completely
    :param ignore_oldexistent:
        If true attribute type names which are in old_entry
        but are not found in new_entry at all are not deleted.
        This is handy for situations where your application
        sets attribute value to '' for deleting an attribute.
        In most cases leave zero.

    :return: List of tuples suitable for
        :py:meth:`ldap:ldap.LDAPObject.modify`.

    This function is the same as :py:func:`ldap:ldap.modlist.modifyModlist`
    except for the following changes:

    * MOD_DELETE/MOD_DELETE used in preference to MOD_REPLACE when updating
      an existing value.
    """
    ignore_attr_types = _list_dict(map(str.lower, (ignore_attr_types or [])))
    modlist: Dict[str, Tuple[str, List[bytes]]] = {}
    attrtype_lower_map = {}
    for a in old_entry.keys():
        attrtype_lower_map[a.lower()] = a
    for attrtype in new_entry.keys():
        attrtype_lower = attrtype.lower()
        if attrtype_lower in ignore_attr_types:
            # This attribute type is ignored
            continue
        # Filter away null-strings
        new_value = list(filter(lambda x: x is not None, new_entry[attrtype]))
        if attrtype_lower in attrtype_lower_map:
            old_value = old_entry.get(attrtype_lower_map[attrtype_lower], [])
            old_value = list(filter(lambda x: x is not None, old_value))
            del attrtype_lower_map[attrtype_lower]
        else:
            old_value = []
        if not old_value and new_value:
            # Add a new attribute to entry
            modlist[attrtype] = (ldap3.MODIFY_ADD, escape_list(new_value))
        elif old_value and new_value:
            # Replace existing attribute
            old_value_dict = _list_dict(old_value)
            new_value_dict = _list_dict(new_value)

            delete_values = []
            for v in old_value:
                if v not in new_value_dict:
                    delete_values.append(v)

            add_values = []
            for v in new_value:
                if v not in old_value_dict:
                    add_values.append(v)

            if len(delete_values) > 0 or len(add_values) > 0:
                modlist[attrtype] = (
                    ldap3.MODIFY_REPLACE, escape_list(new_value))

        elif old_value and not new_value:
            # Completely delete an existing attribute
            modlist[attrtype] = (ldap3.MODIFY_DELETE, [])
    if not ignore_oldexistent:
        # Remove all attributes of old_entry which are not present
        # in new_entry at all
        for a in attrtype_lower_map.keys():
            if a in ignore_attr_types:
                # This attribute type is ignored
                continue
            attrtype = attrtype_lower_map[a]
            modlist[attrtype] = (ldap3.MODIFY_DELETE, [])
    return modlist