def token_handler_str_default(
        token, dispatcher, node, subnode, sourcepath_stack=(None,)):
    """
    Standard token handler that will return the value, ignoring any
    tokens or strings that have been remapped.
    """

    if isinstance(token.pos, int):
        _, lineno, colno = node.getpos(subnode, token.pos)
    else:
        lineno, colno = None, None
    yield StreamFragment(subnode, lineno, colno, None, sourcepath_stack[-1])