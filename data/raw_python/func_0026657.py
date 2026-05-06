def load_command_line_args(clargs=None):
    """Load and parse command-line arguments.

    Arguments
    ---------
    args : str or None
        'Faked' commandline arguments passed to `argparse`.

    Returns
    -------
    args : `argparse.Namespace` object
        Namespace in which settings are stored - default values modified by the
        given command-line arguments.

    """
    import argparse
    git_vers = get_git()

    parser = argparse.ArgumentParser(
        prog='astrocats',
        description='Generate catalogs for astronomical data.')

    parser.add_argument('command', nargs='?', default=None)

    parser.add_argument(
        '--version',
        action='version',
        version='AstroCats v{}, SHA: {}'.format(__version__, git_vers))
    parser.add_argument(
        '--verbose',
        '-v',
        dest='verbose',
        default=False,
        action='store_true',
        help='Print more messages to the screen.')
    parser.add_argument(
        '--debug',
        '-d',
        dest='debug',
        default=False,
        action='store_true',
        help='Print excessive messages to the screen.')
    parser.add_argument(
        '--include-private',
        dest='private',
        default=False,
        action='store_true',
        help='Include private data in import.')
    parser.add_argument(
        '--travis',
        '-t',
        dest='travis',
        default=False,
        action='store_true',
        help='Run import script in test mode for Travis.')
    parser.add_argument(
        '--clone-depth',
        dest='clone_depth',
        default=0,
        type=int,
        help=('When cloning git repos, only clone out to this depth '
              '(default: 0 = all levels).'))
    parser.add_argument(
        '--purge-outputs',
        dest='purge_outputs',
        default=False,
        action='store_true',
        help=('Purge git outputs after cloning.'))
    parser.add_argument(
        '--log',
        dest='log_filename',
        default=None,
        help='Filename to which to store logging information.')

    # If output files should be written or not
    # ----------------------------------------
    write_group = parser.add_mutually_exclusive_group()
    write_group.add_argument(
        '--write',
        action='store_true',
        dest='write_entries',
        default=True,
        help='Write entries to files [default].')
    write_group.add_argument(
        '--no-write',
        action='store_false',
        dest='write_entries',
        default=True,
        help='do not write entries to file.')

    # If previously cleared output files should be deleted or not
    # -----------------------------------------------------------
    delete_group = parser.add_mutually_exclusive_group()
    delete_group.add_argument(
        '--predelete',
        action='store_true',
        dest='delete_old',
        default=True,
        help='Delete all old event files to begin [default].')
    delete_group.add_argument(
        '--no-predelete',
        action='store_false',
        dest='delete_old',
        default=True,
        help='Do not delete all old event files to start.')

    args, sub_clargs = parser.parse_known_args(args=clargs)
    # Print the help information if no command is given
    if args.command is None:
        parser.print_help()
        return None, None

    return args, sub_clargs