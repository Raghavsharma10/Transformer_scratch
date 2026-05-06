def fake_lens_path_set(lens_path, value, obj):
    """
    Simulates R.set with a lens_path since we don't have lens functions
    :param lens_path: Array of string paths
    :param value: The value to set at the lens path
    :param obj: Object containing the given path
    :return: The value at the path or None
    """
    segment = head(lens_path)
    obj_copy = copy.copy(obj)

    def set_array_index(i, v, l):
        # Fill the array with None up to the given index and set the index to v
        try:
            l[i] = v
        except IndexError:
            for _ in range(i - len(l) + 1):
                l.append(None)
            l[i] = v

    if not (length(lens_path) - 1):
        # Done
        new_value = value
    else:
        # Find the value at the path or create a {} or [] at obj[segment]
        found_or_created = item_path_or(
            if_else(
                lambda segment: segment.isnumeric(),
                always([]),
                always({})
            )(head(tail(lens_path))),
            segment,
            obj
        )
        # Recurse on the rest of the path
        new_value = fake_lens_path_set(tail(lens_path), value, found_or_created)

    # Set or replace
    if segment.isnumeric():
        set_array_index(int(segment), new_value, obj_copy)
    else:
        obj_copy[segment] = new_value
    return obj_copy