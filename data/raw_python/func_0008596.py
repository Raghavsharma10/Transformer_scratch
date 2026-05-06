def encode_obj_table(root, plugin):
    """Return list of obj table entries. Converts user-class objects"""
    entries = encode_network(root)

    table_entries = []
    for entry in entries:
        if isinstance(entry, Container):
            assert not hasattr(entry, '__recursion_lock__')
            user_obj_def = plugin.user_objects[entry.class_name]
            attrs = OrderedDict()
            for (key, default) in user_obj_def.defaults.items():
                attrs[key] = entry.get(key, default)
            entry = Container(classID=entry.class_name,
                              length=len(attrs),
                              version=user_obj_def.version,
                              values=attrs.values())
        table_entries.append(entry)
    return table_entries