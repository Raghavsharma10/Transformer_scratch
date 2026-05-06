def find_configured_repository(name):
    """
    Find a version control repository defined by the user in a configuration file.

    :param name: The name of the repository (a string).
    :returns: A :class:`Repository` object.
    :raises: :exc:`~vcs_repo_mgr.exceptions.NoSuchRepositoryError` when the
             given repository name doesn't match any of the configured
             repositories.
    :raises: :exc:`~vcs_repo_mgr.exceptions.AmbiguousRepositoryNameError`
             when the given repository name is ambiguous (i.e. it matches
             multiple repository names).
    :raises: :exc:`~vcs_repo_mgr.exceptions.UnknownRepositoryTypeError` when
             a repository definition with an unknown type is encountered.

    The following configuration files are supported:

    1. ``/etc/vcs-repo-mgr.ini``
    2. ``~/.vcs-repo-mgr.ini``

    Repositories defined in the second file override repositories defined in
    the first. Here is an example of a repository definition:

    .. code-block:: ini

       [vcs-repo-mgr]
       type = git
       local = ~/projects/vcs-repo-mgr
       remote = git@github.com:xolox/python-vcs-repo-mgr.git
       bare = true
       release-scheme = tags
       release-filter = .*

    Three VCS types are currently supported: ``hg`` (``mercurial`` is also
    accepted), ``git`` and ``bzr`` (``bazaar`` is also accepted).
    """
    parser = configparser.RawConfigParser()
    for config_file in [SYSTEM_CONFIG_FILE, USER_CONFIG_FILE]:
        config_file = parse_path(config_file)
        if os.path.isfile(config_file):
            logger.debug("Loading configuration file (%s) ..", format_path(config_file))
            parser.read(config_file)
    matching_repos = [r for r in parser.sections() if normalize_name(name) == normalize_name(r)]
    if not matching_repos:
        msg = "No repositories found matching the name '%s'!"
        raise NoSuchRepositoryError(msg % name)
    elif len(matching_repos) != 1:
        msg = "Multiple repositories found matching the name '%s'! (matches: %s)"
        raise AmbiguousRepositoryNameError(msg % (name, concatenate(map(repr, matching_repos))))
    else:
        kw = {}
        # Get the repository specific options.
        options = dict(parser.items(matching_repos[0]))
        vcs_type = options.get('type', '').lower()
        # Process the `local' directory pathname.
        local_path = options.get('local')
        if local_path:
            # Expand a leading tilde and/or environment variables.
            kw['local'] = parse_path(local_path)
        # Process the `bare' option.
        bare = options.get('bare', None)
        if bare is not None:
            # Default to bare=None but enable configuration
            # file(s) to enforce bare=True or bare=False.
            kw['bare'] = coerce_boolean(bare)
        # Process the `remote', `release_scheme' and `release_filter' options.
        for name in 'remote', 'release-scheme', 'release-filter':
            value = options.get(name)
            if value is not None:
                kw[name.replace('-', '_')] = value
        return repository_factory(vcs_type, **kw)