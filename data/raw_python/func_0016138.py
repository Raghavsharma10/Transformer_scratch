def hide_auth(msg):
    """Remove sensitive information from msg."""
    for pattern, repl in RE_HIDE_AUTH:
        msg = pattern.sub(repl, msg)

    return msg