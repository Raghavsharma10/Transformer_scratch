def get_installed_distributions(local_only=True, skip=('setuptools', 'pip', 'python')):
    """
    Return a list of installed Distribution objects.

    If ``local_only`` is True (default), only return installations
    local to the current virtualenv, if in a virtualenv.

    ``skip`` argument is an iterable of lower-case project names to
    ignore; defaults to ('setuptools', 'pip', 'python'). [FIXME also
    skip virtualenv?]

    """
    if local_only:
        local_test = dist_is_local
    else:
        local_test = lambda d: True
    return [d for d in pkg_resources.working_set if local_test(d) and d.key not in skip]