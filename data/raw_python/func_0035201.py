def token_handler_unobfuscate(
        token, dispatcher, node, subnode, sourcepath_stack=(None,)):
    """
    A token handler that will resolve and return the original identifier
    value.
    """

    original = (
        node.value
        if isinstance(node, Identifier) and node.value != subnode else
        None
    )

    if isinstance(token.pos, int):
        _, lineno, colno = node.getpos(original or subnode, token.pos)
    else:
        lineno, colno = None, None

    yield StreamFragment(
        subnode, lineno, colno, original, sourcepath_stack[-1])