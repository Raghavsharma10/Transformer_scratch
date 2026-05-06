def recurse_tree(app, env, src, dest, excludes, followlinks, force, dryrun, private, suffix):
    """Look for every file in the directory tree and create the corresponding
    ReST files.

    :param app: the sphinx app
    :type app: :class:`sphinx.application.Sphinx`
    :param env: the jinja environment
    :type env: :class:`jinja2.Environment`
    :param src: the path to the python source files
    :type src: :class:`str`
    :param dest: the output directory
    :type dest: :class:`str`
    :param excludes: the paths to exclude
    :type excludes: :class:`list`
    :param followlinks: follow symbolic links
    :type followlinks: :class:`bool`
    :param force: overwrite existing files
    :type force: :class:`bool`
    :param dryrun: do not generate files
    :type dryrun: :class:`bool`
    :param private: include "_private" modules
    :type private: :class:`bool`
    :param suffix: the file extension
    :type suffix: :class:`str`
    """
    # check if the base directory is a package and get its name
    if INITPY in os.listdir(src):
        root_package = src.split(os.path.sep)[-1]
    else:
        # otherwise, the base is a directory with packages
        root_package = None

    toplevels = []
    for root, subs, files in walk(src, followlinks=followlinks):
        # document only Python module files (that aren't excluded)
        py_files = sorted(f for f in files
                          if os.path.splitext(f)[1] in PY_SUFFIXES and  # noqa: W504
                          not is_excluded(os.path.join(root, f), excludes))
        is_pkg = INITPY in py_files
        if is_pkg:
            py_files.remove(INITPY)
            py_files.insert(0, INITPY)
        elif root != src:
            # only accept non-package at toplevel
            del subs[:]
            continue
        # remove hidden ('.') and private ('_') directories, as well as
        # excluded dirs
        if private:
            exclude_prefixes = ('.',)
        else:
            exclude_prefixes = ('.', '_')
        subs[:] = sorted(sub for sub in subs if not sub.startswith(exclude_prefixes) and not
                         is_excluded(os.path.join(root, sub), excludes))
        if is_pkg:
            # we are in a package with something to document
            if subs or len(py_files) > 1 or not \
               shall_skip(app, os.path.join(root, INITPY), private):
                subpackage = root[len(src):].lstrip(os.path.sep).\
                    replace(os.path.sep, '.')
                create_package_file(app, env, root_package, subpackage,
                                    private, dest, suffix, dryrun, force)
                toplevels.append(makename(root_package, subpackage))
        else:
            # if we are at the root level, we don't require it to be a package
            assert root == src and root_package is None
            for py_file in py_files:
                if not shall_skip(app, os.path.join(src, py_file), private):
                    module = os.path.splitext(py_file)[0]
                    create_module_file(app, env, root_package, module, dest, suffix, dryrun, force)
                    toplevels.append(module)
    return toplevels