def decode_obj_table(table_entries, plugin):
    """Return root of obj table. Converts user-class objects"""
    entries = []
    for entry in table_entries:
        if isinstance(entry, Container):
            assert not hasattr(entry, '__recursion_lock__')
            user_obj_def = plugin.user_objects[entry.classID]
            assert entry.version == user_obj_def.version
            entry = Container(class_name=entry.classID,
                              **dict(zip(user_obj_def.defaults.keys(),
                                         entry.values)))
        entries.append(entry)

    return decode_network(entries)