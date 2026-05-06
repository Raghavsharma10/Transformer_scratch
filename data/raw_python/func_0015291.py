def parse_for(control_line):
    """Returns name of loop control variable(s), iteration type (in/word_in) and
    expression to iterate on.

    For example:
    - given "for $i in $foo", returns (['i'], '$foo')
    - given "for ${i} in $(ls $foo)", returns (['i'], '$(ls $foo)')
    - given "for $k, $v in $foo", returns (['k', 'v'], '$foo')
    """
    error = 'For loop call must be in form \'for $var in expression\', got: ' + control_line
    regex = re.compile(r'for\s+(\${?\S+}?)(?:\s*,\s+(\${?\S+}?))?\s+(in|word_in)\s+(\S.+)')
    res = regex.match(control_line)
    if not res:
        raise exceptions.YamlSyntaxError(error)

    groups = res.groups()
    control_vars = []
    control_vars.append(get_var_name(groups[0]))
    if groups[1]:
        control_vars.append(get_var_name(groups[1]))
    iter_type = groups[2]
    expr = groups[3]

    return (control_vars, iter_type, expr)