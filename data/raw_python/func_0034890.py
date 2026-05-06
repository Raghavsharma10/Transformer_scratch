def replace_braces(s, env):
    """Makes tox substitutions to s, with respect to environment env.

    Example
    -------
    >>> replace_braces("echo {posargs:{env:USER:} passed no posargs}")
    "echo andy passed no posargs"

    Note: first "{env:USER:}" is replaced with os.environ.get("USER", ""),
    the "{posargs:andy}" is replaced with "andy" (since no posargs were
    passed).

    """
    def replace(m):
        return _replace_match(m, env)
    for _ in range(DEPTH):
        s = re.sub(r"{[^{}]*}", replace, s)
    return s