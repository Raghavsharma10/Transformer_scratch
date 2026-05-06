def repository_factory(vcs_type, **kw):
    """
    Instantiate a :class:`Repository` object based on the given type and arguments.

    :param vcs_type: One of the strings 'bazaar', 'bzr', 'git', 'hg' or
                     'mercurial' or a subclass of :class:`Repository`.
    :param kw: The keyword arguments to :func:`Repository.__init__()`.
    :returns: A :class:`Repository` object.
    :raises: :exc:`~vcs_repo_mgr.exceptions.UnknownRepositoryTypeError` when
             the given type is unknown.
    """
    # Resolve VCS aliases to Repository subclasses.
    if isinstance(vcs_type, string_types):
        vcs_type = vcs_type.lower()
        for cls in load_backends():
            if vcs_type in cls.ALIASES:
                vcs_type = cls
                break
    # Make sure we have a valid repository type to work with.
    if not (isinstance(vcs_type, type) and issubclass(vcs_type, Repository)):
        raise UnknownRepositoryTypeError("Unknown VCS repository type! (%r)" % vcs_type)
    # Generate a cache key that we will use to avoid constructing duplicates.
    cache_key = tuple('%s=%s' % (k, v) for k, v in sorted(kw.items()))
    logger.debug("Generated repository cache key: %r", cache_key)
    if cache_key in loaded_repositories:
        logger.debug("Repository previously constructed, returning cached instance ..")
    else:
        logger.debug("Repository not yet constructed, creating new instance ..")
        loaded_repositories[cache_key] = vcs_type(**kw)
    return loaded_repositories[cache_key]