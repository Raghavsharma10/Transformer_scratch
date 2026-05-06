def status(c):
    """
    Print current release (version, changelog, tag, etc) status.

    Doubles as a subroutine, returning the return values from its inner call to
    ``_converge`` (an ``(actions, state)`` two-tuple of Lexicons).
    """
    # TODO: wants some holistic "you don't actually HAVE any changes to
    # release" final status - i.e. all steps were at no-op status.
    actions, state = _converge(c)
    table = []
    # NOTE: explicit 'sensible' sort (in rough order of how things are usually
    # modified, and/or which depend on one another, e.g. tags are near the end)
    for component in "changelog version tag".split():
        table.append((component.capitalize(), actions[component].value))
    print(tabulate(table))
    return actions, state