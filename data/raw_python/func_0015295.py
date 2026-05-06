def assign_variable(variable, log_res, res, kwargs):
    """Assigns given result (resp. logical result and result) to a variable
    (resp. to two variables). log_res and res are already computed result
    of an exec/input section. For example:

    $foo~: $spam and $eggs
    $log_res, $foo~: $spam and $eggs

    $foo:
      some: struct
    $log_res, $foo:
      some:
        other:
          struct

    Args:
        variable: variable (or two variables separated by ",") to assign to
        log_res: logical result of evaluated section
        res: result of evaluated section

    Raises:
        YamlSyntaxError: if there are more than two variables
    """
    if variable.endswith('~'):
        variable = variable[:-1]
    comma_count = variable.count(',')
    if comma_count > 1:
        raise exceptions.YamlSyntaxError('Max two variables allowed on left side.')

    if comma_count == 1:
        var1, var2 = map(lambda v: get_var_name(v), variable.split(','))
        kwargs[var1] = log_res
    else:
        var2 = get_var_name(variable)
    kwargs[var2] = res
    return log_res, res