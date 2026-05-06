def find_value_in_object(attr, obj):
    """Return values for any key coincidence with attr in obj or any other
    nested dict.
    """

    # Carry on inspecting inside the list or tuple
    if isinstance(obj, (collections.Iterator, list)):
        for item in obj:
            yield from find_value_in_object(attr, item)

    # Final object (dict or entity) inspect inside
    elif isinstance(obj, collections.Mapping):

        # If result is found, inspect inside and return inner results
        if attr in obj:

            # If it is iterable, just return the inner elements (avoid nested
            # lists)
            if isinstance(obj[attr], (collections.Iterator, list)):
                for item in obj[attr]:
                    yield item

            # If not, return just the objects
            else:
                yield obj[attr]

        # Carry on inspecting inside the object
        for item in obj.values():
            if item:
                yield from find_value_in_object(attr, item)