def make_stack(env, stage, segment):
    """For each transform segment, create the code in the try/except block with the
    assignements for pipes in the segment """

    import string
    import random
    from ambry.valuetype import ValueType

    column = segment['column']

    def make_line(column, t):
        preamble = []

        line_t = "v = {} # {}"

        if isinstance(t, type) and issubclass(t, ValueType):  # A valuetype class, from the datatype column.

            try:
                cc, fl = calling_code(t, t.__name__), file_loc()
            except TypeError:
                cc, fl = "{}(v)".format(t.__name__), file_loc()

            preamble.append("{} = resolve_value_type('{}') # {}".format(t.__name__, t.vt_code, fl))

        elif isinstance(t, type):  # A python type, from the datatype columns.
            cc, fl= "parse_{}(v, header_d)".format(t.__name__), file_loc()

        elif callable(env.get(t)):  # Transform function
            cc, fl = calling_code(env.get(t), t), file_loc()

        else:  # A transform generator, or python code.

            rnd = (''.join(random.choice(string.ascii_lowercase) for _ in range(6)))

            name = 'tg_{}_{}_{}'.format(column.name, stage, rnd)
            try:
                a, b, fl = rewrite_tg(env, name, t)
            except CodeGenError as e:
                raise CodeGenError("Failed to re-write pipe code '{}' in column '{}.{}': {} "
                                   .format(t, column.table.name, column.name, e))

            cc = str(a)

            if b:
                preamble.append("{} = {} # {}".format(name, b, fl))

        line = line_t.format(cc, fl)

        return line, preamble

    preamble = []

    try_lines = []

    for t in [segment['init'], segment['datatype']] + segment['transforms']:

        if not t:
            continue

        line, col_preamble = make_line(column, t)

        preamble += col_preamble
        try_lines.append(line)

    exception = None
    if segment['exception']:
        exception, col_preamble = make_line(column, segment['exception'])

    if len(try_lines) == 0:
        try_lines.append('pass # Empty pipe segment')

    assert len(try_lines) > 0, column.name

    return preamble, try_lines, exception