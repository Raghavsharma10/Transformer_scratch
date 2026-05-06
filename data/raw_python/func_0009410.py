def _pythonized_comments(tokens):
    """
    Similar to tokens but converts strings after a colon (:) to comments.
    """
    is_after_colon = True
    for token_type, token_text in tokens:
        if is_after_colon and (token_type in pygments.token.String):
            token_type = pygments.token.Comment
        elif token_text == ':':
            is_after_colon = True
        elif token_type not in pygments.token.Comment:
            is_whitespace = len(token_text.rstrip(' \f\n\r\t')) == 0
            if not is_whitespace:
                is_after_colon = False
        yield token_type, token_text