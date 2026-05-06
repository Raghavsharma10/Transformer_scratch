def do_for(parser, token):
    '''
    {% for a, b, c in iterable %}

    {% endfor %}

    We create the structure:

    with ContextWrapper(context) as context:
        for a, b, c in iterable:
            context.update(a=a, b=b, c=c)
            ...

    If there is a {% empty %} clause, we create:

    if iterable:
        { above code }
    else:
        { empty clause }
    '''
    code = ast.parse('for %s: pass' % token, mode='exec')

    # Grab the ast.For node
    loop = code.body[0]
    # Wrap its source iterable
    loop.iter = visitor.visit(loop.iter)

    # Get the body of the loop
    body, end = parser.parse_nodes_until('endfor', 'empty')

    # Build a list of target variable names
    if isinstance(loop.target, ast.Tuple):
        targets = [elt.id for elt in loop.target.elts]
    else:
        targets = [loop.target.id]

    kwargs = [
        ast.keyword(arg=elt, value=_a.Name(elt))
        for elt in targets
    ]

    # Insert our update call at the start of the loop body
    body.insert(0, ast.Expr(value=_a.Call(
        _a.Attribute(_a.Name('context'), 'update'),
        keywords=kwargs
    )))
    loop.body = body

    node = _create_with_scope([loop], [])

    if end == 'empty':
        # Now we wrap our for block in:
        # if loop.iter:
        # else:
        empty, _ = parser.parse_nodes_until('endfor')

        node = ast.If(
            test=loop.iter,
            body=[node],
            orelse=empty
        )

    return node