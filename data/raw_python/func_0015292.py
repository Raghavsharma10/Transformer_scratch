def get_for_control_var_and_eval_expr(comm_type, kwargs):
    """Returns tuple that consists of control variable name and iterable that is result
    of evaluated expression of given for loop.

    For example:
    - given 'for $i in $(echo "foo bar")' it returns (['i'], ['foo', 'bar'])
    - given 'for $i, $j in $foo' it returns (['i', 'j'], [('foo', 'bar')])
    """
    # let possible exceptions bubble up
    control_vars, iter_type, expression = parse_for(comm_type)
    eval_expression = evaluate_expression(expression, kwargs)[1]

    iterval = []
    if len(control_vars) == 2:
        if not isinstance(eval_expression, dict):
            raise exceptions.YamlSyntaxError('Can\'t expand {t} to two control variables.'.
                                             format(t=type(eval_expression)))
        else:
            iterval = list(eval_expression.items())
    elif isinstance(eval_expression, six.string_types):
        if iter_type == 'word_in':
            iterval = eval_expression.split()
        else:
            iterval = eval_expression
    else:
        iterval = eval_expression

    return control_vars, iterval