def all_pass_dict(f, dct):
    """
        Returns true if all dct values pass f
    :param f: binary lambda predicate
    :param dct:
    :return: True or false
    """
    return all(map_with_obj_to_values(
        lambda key, value: f(key, value),
        dct
    ))