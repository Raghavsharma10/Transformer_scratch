def main():
    """Command line interface for the ``rotate-backups`` program."""
    coloredlogs.install(syslog=True)
    # Command line option defaults.
    rotation_scheme = {}
    kw = dict(include_list=[], exclude_list=[])
    parallel = False
    use_sudo = False
    # Internal state.
    selected_locations = []
    # Parse the command line arguments.
    try:
        options, arguments = getopt.getopt(sys.argv[1:], 'M:H:d:w:m:y:I:x:jpri:c:r:uC:nvqh', [
            'minutely=', 'hourly=', 'daily=', 'weekly=', 'monthly=', 'yearly=',
            'include=', 'exclude=', 'parallel', 'prefer-recent', 'relaxed',
            'ionice=', 'config=', 'use-sudo', 'dry-run', 'removal-command=',
            'verbose', 'quiet', 'help',
        ])
        for option, value in options:
            if option in ('-M', '--minutely'):
                rotation_scheme['minutely'] = coerce_retention_period(value)
            elif option in ('-H', '--hourly'):
                rotation_scheme['hourly'] = coerce_retention_period(value)
            elif option in ('-d', '--daily'):
                rotation_scheme['daily'] = coerce_retention_period(value)
            elif option in ('-w', '--weekly'):
                rotation_scheme['weekly'] = coerce_retention_period(value)
            elif option in ('-m', '--monthly'):
                rotation_scheme['monthly'] = coerce_retention_period(value)
            elif option in ('-y', '--yearly'):
                rotation_scheme['yearly'] = coerce_retention_period(value)
            elif option in ('-I', '--include'):
                kw['include_list'].append(value)
            elif option in ('-x', '--exclude'):
                kw['exclude_list'].append(value)
            elif option in ('-j', '--parallel'):
                parallel = True
            elif option in ('-p', '--prefer-recent'):
                kw['prefer_recent'] = True
            elif option in ('-r', '--relaxed'):
                kw['strict'] = False
            elif option in ('-i', '--ionice'):
                value = validate_ionice_class(value.lower().strip())
                kw['io_scheduling_class'] = value
            elif option in ('-c', '--config'):
                kw['config_file'] = parse_path(value)
            elif option in ('-u', '--use-sudo'):
                use_sudo = True
            elif option in ('-n', '--dry-run'):
                logger.info("Performing a dry run (because of %s option) ..", option)
                kw['dry_run'] = True
            elif option in ('-C', '--removal-command'):
                removal_command = shlex.split(value)
                logger.info("Using custom removal command: %s", removal_command)
                kw['removal_command'] = removal_command
            elif option in ('-v', '--verbose'):
                coloredlogs.increase_verbosity()
            elif option in ('-q', '--quiet'):
                coloredlogs.decrease_verbosity()
            elif option in ('-h', '--help'):
                usage(__doc__)
                return
            else:
                assert False, "Unhandled option! (programming error)"
        if rotation_scheme:
            logger.verbose("Rotation scheme defined on command line: %s", rotation_scheme)
        if arguments:
            # Rotation of the locations given on the command line.
            location_source = 'command line arguments'
            selected_locations.extend(coerce_location(value, sudo=use_sudo) for value in arguments)
        else:
            # Rotation of all configured locations.
            location_source = 'configuration file'
            selected_locations.extend(
                location for location, rotation_scheme, options
                in load_config_file(configuration_file=kw.get('config_file'), expand=True)
            )
        # Inform the user which location(s) will be rotated.
        if selected_locations:
            logger.verbose("Selected %s based on %s:",
                           pluralize(len(selected_locations), "location"),
                           location_source)
            for number, location in enumerate(selected_locations, start=1):
                logger.verbose(" %i. %s", number, location)
        else:
            # Show the usage message when no directories are given nor configured.
            logger.verbose("No location(s) to rotate selected.")
            usage(__doc__)
            return
    except Exception as e:
        logger.error("%s", e)
        sys.exit(1)
    # Rotate the backups in the selected directories.
    program = RotateBackups(rotation_scheme, **kw)
    if parallel:
        program.rotate_concurrent(*selected_locations)
    else:
        for location in selected_locations:
            program.rotate_backups(location)