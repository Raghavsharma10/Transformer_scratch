def use(parser, token):
    '''
    Counterpart to `macro`, lets you render any block/macro in place.
    '''

    args, kwargs = parser.parse_args(token)
    assert isinstance(args[0], ast.Str), \
        'First argument to "include" tag must be a string'
    name = args[0].s

    action = ast.YieldFrom(
        value=_a.Call(_a.Attribute(_a.Name('self'), name), [
            _a.Name('context'),
        ])
    )

    if kwargs:
        kwargs = _wrap_kwargs(kwargs)
        return _create_with_scope([ast.Expr(value=action)], kwargs)

    return action