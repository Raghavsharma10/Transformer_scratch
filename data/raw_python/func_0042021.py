def macro(parser, token):
    '''
    Works just like block, but does not render.
    '''
    name = token.strip()
    parser.build_method(name, endnodes=['endmacro'])
    return ast.Yield(value=ast.Str(s=''))