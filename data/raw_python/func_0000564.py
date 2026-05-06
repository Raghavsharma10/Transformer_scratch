def node_is_result_assignment(node: ast.AST) -> bool:
    """
    Args:
        node: An ``ast`` node.

    Returns:
        bool: ``node`` corresponds to the code ``result =``, assignment to the
        ``result `` variable.

    Note:
        Performs a very weak test that the line starts with 'result =' rather
        than testing the tokens.
    """
    # `.first_token` is added by asttokens
    token = node.first_token  # type: ignore
    return token.line.strip().startswith('result =')