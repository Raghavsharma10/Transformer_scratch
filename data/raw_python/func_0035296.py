def toplevel_packages():
    """ Get package list, without sub-packages.
    """
    packages = set(easy.options.setup.packages)
    for pkg in list(packages):
        packages -= set(p for p in packages if str(p).startswith(pkg + '.'))
    return list(sorted(packages))