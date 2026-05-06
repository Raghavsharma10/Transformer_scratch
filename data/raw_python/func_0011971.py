def generate(app, src, dest, exclude=[], followlinks=False,
             force=False, dryrun=False, private=False, suffix='rst',
             template_dirs=None):
    """Generage the rst files

    Raises an :class:`OSError` if the source path is not a directory.

    :param app: the sphinx app
    :type app: :class:`sphinx.application.Sphinx`
    :param src: path to python source files
    :type src: :class:`str`
    :param dest: output directory
    :type dest: :class:`str`
    :param exclude: list of paths to exclude
    :type exclude: :class:`list`
    :param followlinks: follow symbolic links
    :type followlinks: :class:`bool`
    :param force: overwrite existing files
    :type force: :class:`bool`
    :param dryrun: do not create any files
    :type dryrun: :class:`bool`
    :param private: include \"_private\" modules
    :type private: :class:`bool`
    :param suffix: file suffix
    :type suffix: :class:`str`
    :param template_dirs: directories to search for user templates
    :type template_dirs: None | :class:`list`
    :returns: None
    :rtype: None
    :raises: OSError
    """
    suffix = suffix.strip('.')
    if not os.path.isdir(src):
        raise OSError("%s is not a directory" % src)
    if not os.path.isdir(dest) and not dryrun:
        os.makedirs(dest)
    src = os.path.normpath(os.path.abspath(src))
    exclude = normalize_excludes(exclude)
    loader = make_loader(template_dirs)
    env = make_environment(loader)
    recurse_tree(app, env, src, dest, exclude, followlinks, force, dryrun, private, suffix)