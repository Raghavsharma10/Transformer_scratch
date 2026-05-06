def load_config_file(configuration_file=None, expand=True):
    """
    Load a configuration file with backup directories and rotation schemes.

    :param configuration_file: Override the pathname of the configuration file
                               to load (a string or :data:`None`).
    :param expand: :data:`True` to expand filename patterns to their matches,
                   :data:`False` otherwise.
    :returns: A generator of tuples with four values each:

              1. An execution context created using :mod:`executor.contexts`.
              2. The pathname of a directory with backups (a string).
              3. A dictionary with the rotation scheme.
              4. A dictionary with additional options.
    :raises: :exc:`~exceptions.ValueError` when `configuration_file` is given
             but doesn't exist or can't be loaded.

    This function is used by :class:`RotateBackups` to discover user defined
    rotation schemes and by :mod:`rotate_backups.cli` to discover directories
    for which backup rotation is configured. When `configuration_file` isn't
    given :class:`~update_dotdee.ConfigLoader` is used to search for
    configuration files in the following locations:

    - ``/etc/rotate-backups.ini`` and ``/etc/rotate-backups.d/*.ini``
    - ``~/.rotate-backups.ini`` and ``~/.rotate-backups.d/*.ini``
    - ``~/.config/rotate-backups.ini`` and ``~/.config/rotate-backups.d/*.ini``

    All of the available configuration files are loaded in the order given
    above, so that sections in user-specific configuration files override
    sections by the same name in system-wide configuration files.
    """
    expand_notice_given = False
    if configuration_file:
        loader = ConfigLoader(available_files=[configuration_file], strict=True)
    else:
        loader = ConfigLoader(program_name='rotate-backups', strict=False)
    for section in loader.section_names:
        items = dict(loader.get_options(section))
        context_options = {}
        if coerce_boolean(items.get('use-sudo')):
            context_options['sudo'] = True
        if items.get('ssh-user'):
            context_options['ssh_user'] = items['ssh-user']
        location = coerce_location(section, **context_options)
        rotation_scheme = dict((name, coerce_retention_period(items[name]))
                               for name in SUPPORTED_FREQUENCIES
                               if name in items)
        options = dict(include_list=split(items.get('include-list', '')),
                       exclude_list=split(items.get('exclude-list', '')),
                       io_scheduling_class=items.get('ionice'),
                       strict=coerce_boolean(items.get('strict', 'yes')),
                       prefer_recent=coerce_boolean(items.get('prefer-recent', 'no')))
        # Don't override the value of the 'removal_command' property unless the
        # 'removal-command' configuration file option has a value set.
        if items.get('removal-command'):
            options['removal_command'] = shlex.split(items['removal-command'])
        # Expand filename patterns?
        if expand and location.have_wildcards:
            logger.verbose("Expanding filename pattern %s on %s ..", location.directory, location.context)
            if location.is_remote and not expand_notice_given:
                logger.notice("Expanding remote filename patterns (may be slow) ..")
                expand_notice_given = True
            for match in sorted(location.context.glob(location.directory)):
                if location.context.is_directory(match):
                    logger.verbose("Matched directory: %s", match)
                    expanded = Location(context=location.context, directory=match)
                    yield expanded, rotation_scheme, options
                else:
                    logger.verbose("Ignoring match (not a directory): %s", match)
        else:
            yield location, rotation_scheme, options