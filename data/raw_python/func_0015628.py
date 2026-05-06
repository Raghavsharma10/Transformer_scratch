def _construct_target_list(targets):
    """Create a list of TargetEntry items from a list of tuples in the form (target, flags, info)

    The list can also contain existing TargetEntry items in which case the existing entry
    is re-used in the return list.
    """
    target_entries = []
    for entry in targets:
        if not isinstance(entry, Gtk.TargetEntry):
            entry = Gtk.TargetEntry.new(*entry)
        target_entries.append(entry)
    return target_entries