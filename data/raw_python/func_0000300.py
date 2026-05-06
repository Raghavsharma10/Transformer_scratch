def get_diff(source, dest):
    """Get the diff between two records list in this order:
        - to_create
        - to_delete
    """
    # First build a dict from the lists, with the ID as the key.
    source_dict = {record['id']: record for record in source}
    dest_dict = {record['id']: record for record in dest}

    source_keys = set(source_dict.keys())
    dest_keys = set(dest_dict.keys())
    to_create = source_keys - dest_keys
    to_delete = dest_keys - source_keys
    to_update = set()

    to_check = source_keys - to_create - to_delete

    for record_id in to_check:
        # Make sure to remove properties that are part of kinto
        # records and not amo records.
        # Here we will compare the record properties ignoring:
        # ID, last_modified and enabled.
        new = canonical_json(source_dict[record_id])
        old = canonical_json(dest_dict[record_id])
        if new != old:
            to_update.add(record_id)

    return ([source_dict[k] for k in to_create],
            [source_dict[k] for k in to_update],
            [dest_dict[k] for k in to_delete])