def process_params(mod_id, params, type_params):
    """
    Takes as input a dictionary of parameters defined on a module and the
    information about the required parameters defined on the corresponding
    module type. Validatates that are required parameters were supplied and
    fills any missing parameters with their default values from the module
    type. Returns a nested dictionary of the same format as the `type_params`
    but with an additional key `value` on each inner dictionary that gives the
    value of that parameter for this specific module
    """
    res = {}
    for param_name, param_info in type_params.items():
        val = params.get(param_name, param_info.get("default", None))
        # Check against explicit None (param could be explicitly False)
        if val is not None:
            param_res = dict(param_info)
            param_res["value"] = val
            res[param_name] = param_res
        elif type_params.get("required", False):
            raise ValueError(
                'Required parameter "{}" is not defined for module '
                '"{}"'.format(param_name, mod_id)
            )
    return res