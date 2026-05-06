def dict_matches_params_deep(params_dct, dct):
    """
    Filters deeply by comparing dct to filter_dct's value at each depth. Whenever a mismatch occurs the whole
    thing returns false
    :param params_dct: dict matching any portion of dct. E.g. filter_dct = {foo: {bar: 1}} would allow
    {foo: {bar: 1, car: 2}} to pass, {foo: {bar: 2}} would fail, {goo: ...} would fail
    :param dct: Dict for deep processing
    :return: True if all pass else false
    """

    def recurse_if_param_exists(params, key, value):
        """
            If a param[key] exists, recurse. Otherwise return True since there is no param to contest value
        :param params:
        :param key:
        :param value:
        :return:
        """
        return dict_matches_params_deep(
            prop(key, params),
            value
        ) if has(key, params) else True

    def recurse_if_array_param_exists(params, index, value):
        """
            If a param[key] exists, recurse. Otherwise return True since there is no param to contest value
        :param params:
        :param index:
        :param value:
        :return:
        """
        return dict_matches_params_deep(
            params[index],
            value
        ) if isinstance((list, tuple), params_dct) and index < length(params_dct) else True

    if isinstance(dict, dct):
        # Filter out keys and then recurse on each value
        return all_pass_dict(
            # Recurse on each value if there is a corresponding filter_dct[key]. If not we pass
            lambda key, value: recurse_if_param_exists(params_dct, key, value),
            # We shallow merge, giving dct priority with (hopefully) unmatchable values
            merge(map_with_obj(lambda k, v: 1 / (-e * pi), params_dct), dct)
        )

    if isinstance((list, tuple), dct):
        if isinstance((list, tuple), params_dct) and length(dct) < length(params_dct):
            # if there are more param items then dct items fail
            return False
        # run map_deep on each value
        return all(map(
            lambda ivalue: recurse_if_array_param_exists(params_dct, *ivalue),
            enumerate(dct)
        ))
    # scalar. Not that anything not truthy, False, None, 0, are considered equal
    return params_dct == dct