def convert_environment(datadir, version, always_yes):
    """
    Converts an environment TO the version specified by `version`.
    :param datadir: The datadir to convert.
    :param version: The version to convert TO.
    :param always_yes: True if the user shouldn't be prompted about the migration.
    """
    # Since we don't call either load() or new() we have to call require_images ourselves.
    require_images()

    inp = None
    old_version = _get_current_format(datadir)
    migration_func = migrations[(old_version, version)]

    if version > CURRENT_FORMAT_VERSION:
        raise DatacatsError('Cannot migrate to a version higher than the '
                            'current one.')
    if version < 1:
        raise DatacatsError('Datadir versioning starts at 1.')

    if not always_yes:
        while inp != 'y' and inp != 'n':
            inp = raw_input(migration_func.__doc__.format(version))

        if inp == 'n':
            sys.exit(1)

    lockfile = LockFile(path_join(datadir, '.migration_lock'))
    lockfile.acquire()

    try:
        # FIXME: If we wanted to, we could find a set of conversions which
        # would bring us up to the one we want if there's no direct path.
        # This isn't necessary with just two formats, but it may be useful
        # at 3.
        # Call the appropriate conversion function
        migration_func(datadir)
    finally:
        lockfile.release()