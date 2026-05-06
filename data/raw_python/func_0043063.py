def build_dot_value(key, value):
    """Build new dictionaries based off of the dot notation key.

    For example, if a key were 'x.y.z' and the value was 'foo',
    we would expect a return value of: ('x', {'y': {'z': 'foo'}})

    Args:
        key (str): The key to build a dictionary off of.
        value: The value associated with the dot notation key.

    Returns:
        tuple: A 2-tuple where the first element is the key of
            the outermost scope (e.g. left-most in the dot
            notation key) and the value is the constructed value
            for that key (e.g. a dictionary)
    """
    # if there is no nesting in the key (as specified by the
    # presence of dot notation), then the key/value pair here
    # are the final key value pair.
    if key.count('.') == 0:
        return key, value

    # otherwise, we will need to construct as many dictionaries
    # as there are dot components to hold the value.
    final_value = value
    reverse_split = key.split('.')[::-1]
    end = len(reverse_split) - 1
    for idx, k in enumerate(reverse_split):
        if idx == end:
            return k, final_value
        final_value = {k: final_value}