def preload(python_data: LdapObject, database: Optional[Database] = None) -> LdapObject:
    """ Preload all NotLoaded fields in LdapObject. """

    changes = {}

    # Load objects within lists.
    def preload_item(value: Any) -> Any:
        if isinstance(value, NotLoaded):
            return value.load(database)
        else:
            return value

    for name in python_data.keys():
        value_list = python_data.get_as_list(name)

        # Check for errors.
        if isinstance(value_list, NotLoadedObject):
            raise RuntimeError(f"{name}: Unexpected NotLoadedObject outside list.")

        elif isinstance(value_list, NotLoadedList):
            value_list = value_list.load(database)

        else:
            if any(isinstance(v, NotLoadedList) for v in value_list):
                raise RuntimeError(f"{name}: Unexpected NotLoadedList in list.")
            elif any(isinstance(v, NotLoadedObject) for v in value_list):
                value_list = [preload_item(value) for value in value_list]
            else:
                value_list = None

        if value_list is not None:
            changes[name] = value_list

    return python_data.merge(changes)