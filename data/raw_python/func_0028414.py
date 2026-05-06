def flatten_top_level_keys(data, top_level_keys):
    """ Helper method to flatten a nested dict of dicts (one level)

        Example:
            {'a': {'b': 'bbb'}} becomes {'a_-_b': 'bbb'}

            The separator '_-_' gets formatted later for the column headers

        Args:
            data: the dict to flatten
            top_level_keys: a list of the top level keys to flatten ('a' in the example above)
    """
    flattened_data = {}

    for top_level_key in top_level_keys:
        if data[top_level_key] is None:
            flattened_data[top_level_key] = None
        else:
            for key in data[top_level_key]:
                flattened_data['{}_-_{}'.format(top_level_key, key)] = data[top_level_key][key]

    return flattened_data