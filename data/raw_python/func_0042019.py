def _create_with_scope(body, kwargs):
    '''
    Helper function to wrap a block in a scope stack:

    with ContextScope(context, **kwargs) as context:
        ... body ...
    '''
    return ast.With(
        items=[
            ast.withitem(
                context_expr=_a.Call(
                    _a.Name('ContextScope'),
                    [_a.Name('context')],
                    keywords=kwargs,
                ),
                optional_vars=_a.Name('context', ctx=ast.Store())
            ),
        ],
        body=body,
    )