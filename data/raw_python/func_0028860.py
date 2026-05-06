def unflatten_dct(obj):
    """
    Undoes the work of flatten_dict
    @param {Object} obj 1-D object in the form returned by flattenObj
    @returns {Object} The original 
    :param obj: 
    :return: 
    """

    def reduce_func(accum, key_string_and_value):
        key_string = key_string_and_value[0]
        value = key_string_and_value[1]
        item_key_path = key_string_to_lens_path(key_string)
        # All but the last segment gives us the item container len
        container_key_path = init(item_key_path)
        container = unless(
            # If the path has any length (not []) and the value is set, don't do anything
            both(always(length(container_key_path)), fake_lens_path_view(container_key_path)),
            # Else we are at the top level, so use the existing accum or create a [] or {}
            # depending on if our item key is a number or not
            lambda x: default_to(
                if_else(
                    lambda segment: segment.isnumeric(),
                    always([]),
                    always({})
                )(head(item_key_path))
            )(x)
        )(accum)
        # Finally set the container at the itemLensPath
        return fake_lens_path_set(
            item_key_path,
            value,
            container
        )

    return compose(
        reduce(
            reduce_func,
            # null initial value
            None
        ),
        to_pairs
    )(obj)