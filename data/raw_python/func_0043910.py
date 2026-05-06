def coerce_repository(value, context=None):
    """
    Convert a string (taken to be a repository name or location) to a :class:`Repository` object.

    :param value: The name or location of a repository (a string) or a
                  :class:`Repository` object.
    :param context: An execution context created by :mod:`executor.contexts`
                    (defaults to :class:`executor.contexts.LocalContext`).
    :returns: A :class:`Repository` object.
    :raises: :exc:`~exceptions.ValueError` when the given value is not a string
             or a :class:`Repository` object or if the value is a string but
             doesn't match the name of any configured repository and also can't
             be parsed as the location of a repository.

    The :func:`coerce_repository()` function creates :class:`Repository` objects:

    1. If the value is already a :class:`Repository` object it is returned to
       the caller untouched.
    2. If the value is accepted by :func:`find_configured_repository()` then
       the resulting :class:`Repository` object is returned.
    3. If the value is a string that starts with a known VCS type prefix (e.g.
       ``hg+https://bitbucket.org/ianb/virtualenv``) the prefix is removed from
       the string and a :class:`Repository` object is returned:

       - If the resulting string points to an existing local directory it will
         be used to set :attr:`~Repository.local`.
       - Otherwise the resulting string is used to set
         :attr:`~Repository.remote`.
    4. If the value is a string pointing to an existing local directory, the
       VCS type is inferred from the directory's contents and a
       :class:`Repository` object is returned whose :attr:`~Repository.local`
       property is set to the local directory.
    5. If the value is a string that ends with ``.git`` (a common idiom for git
       repositories) a :class:`Repository` object is returned:

       - If the value points to an existing local directory it will be used to
         set :attr:`~Repository.local`.
       - Otherwise the value is used to set :attr:`~Repository.remote`.
    """
    # Coerce the context argument.
    context = context or LocalContext()
    # Repository objects pass through untouched.
    if isinstance(value, Repository):
        return value
    # We expect a string with a name or URL.
    if not isinstance(value, string_types):
        msg = "Expected string or Repository object as argument, got %s instead!"
        raise ValueError(msg % type(value))
    # If the string matches the name of a configured repository we'll return that.
    try:
        return find_configured_repository(value)
    except NoSuchRepositoryError:
        pass
    # Parse and try to resolve the VCS type prefix.
    vcs_type, _, location = value.partition('+')
    if vcs_type and location:
        kw = {
            'context': context,
            'local' if context.exists(location) else 'remote': location,
        }
        try:
            return repository_factory(vcs_type, **kw)
        except UnknownRepositoryTypeError:
            pass
    # Try to infer the type of an existing local repository.
    for cls in load_backends():
        if cls.contains_repository(context, value):
            return repository_factory(cls, context=context, local=value)
    # Check for locations that end with `.git' (a common idiom for remote
    # git repositories) even if the location isn't prefixed with `git+'.
    if value.endswith('.git'):
        from vcs_repo_mgr.backends.git import GitRepo
        return repository_factory(GitRepo, **{
            'context': context,
            'local' if context.exists(value) else 'remote': value,
        })
    # If all else fails, at least give a clear explanation of the problem.
    msg = ("The string %r doesn't match the name of any configured repository"
           " and it also can't be parsed as the location of a remote"
           " repository! (maybe you forgot to prefix the type?)")
    raise ValueError(msg % value)