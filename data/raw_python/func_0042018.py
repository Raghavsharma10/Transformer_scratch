def do_super(parser, token):
    '''
    Access the parent templates block.

    {% super name %}
    '''
    name = token.strip()
    return ast.YieldFrom(
        value=_a.Call(_a.Attribute(_a.Call(_a.Name('super')), name), [
            # _a.Attribute(_a.Name('context'), 'parent'),
            _a.Name('context'),
        ])
    )