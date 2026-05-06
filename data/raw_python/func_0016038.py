def duplicate_ids(instance):
    """Ensure objects with duplicate IDs have different `modified` timestamps.
    """
    if instance['type'] != 'bundle' or 'objects' not in instance:
        return

    unique_ids = {}
    for obj in instance['objects']:
        if 'id' not in obj or 'modified' not in obj:
            continue
        elif obj['id'] not in unique_ids:
            unique_ids[obj['id']] = obj['modified']
        elif obj['modified'] == unique_ids[obj['id']]:
            yield JSONError("Duplicate ID '%s' has identical `modified` timestamp."
                            " If they are different versions of the same object, "
                            "they should have different `modified` properties."
                            % obj['id'], instance['id'], 'duplicate-ids')