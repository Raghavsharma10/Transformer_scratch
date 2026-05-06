def sync(obj, engine):
    """Mark the object as having been persisted at least once.

    Store the latest snapshot of all marked values."""
    snapshot = Condition()
    # Only expect values (or lack of a value) for columns that have been explicitly set
    for column in sorted(_obj_tracking[obj]["marked"], key=lambda col: col.dynamo_name):
        value = getattr(obj, column.name, None)
        value = engine._dump(column.typedef, value)
        condition = column == value
        # The renderer shouldn't try to dump the value again.
        # We're dumping immediately in case the value is mutable,
        # such as a set or (many) custom data types.
        condition.dumped = True
        snapshot &= condition
    _obj_tracking[obj]["snapshot"] = snapshot