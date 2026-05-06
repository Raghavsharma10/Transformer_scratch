def main():
    """Entry point when called on the command-line.
    """
    # Locale
    locale.setlocale(locale.LC_ALL, '')

    # Encoding for output streams
    if str == bytes:  # PY2
        writer = codecs.getwriter(locale.getpreferredencoding())
        o_stdout, o_stderr = sys.stdout, sys.stderr
        sys.stdout = writer(sys.stdout)
        sys.stdout.buffer = o_stdout
        sys.stderr = writer(sys.stderr)
        sys.stderr.buffer = o_stderr
    else:  # PY3
        sys.stdin = sys.stdin.buffer

    # Parses command-line

    # Runtime to setup
    def add_runtime_option(opt):
        opt.add_argument(
            '-r', '--runtime', action='store',
            help="runtime to deploy on the server if the queue doesn't exist. "
                 "If unspecified, will auto-detect what is appropriate, and "
                 "fallback on 'default'.")

    # Destination selection
    def add_destination_option(opt):
        opt.add_argument('destination', action='store',
                         help="Machine to SSH into; [user@]host[:port]")
        opt.add_argument('--queue', action='store', default=DEFAULT_TEJ_DIR,
                         help="Directory for tej's files")

    # Root parser
    parser = argparse.ArgumentParser(
        description="Trivial Extensible Job-submission")
    parser.add_argument('--version', action='version',
                        version="tej version %s" % tej_version)
    parser.add_argument('-v', '--verbose', action='count', default=1,
                        dest='verbosity',
                        help="augments verbosity level")
    subparsers = parser.add_subparsers(title="commands", metavar='')

    # Setup action
    parser_setup = subparsers.add_parser(
        'setup',
        help="Sets up tej on a remote machine")
    add_destination_option(parser_setup)
    add_runtime_option(parser_setup)
    parser_setup.add_argument('--make-link', action='append',
                              dest='make_link')
    parser_setup.add_argument('--make-default-link', action='append_const',
                              dest='make_link', const=DEFAULT_TEJ_DIR)
    parser_setup.add_argument('--force', action='store_true')
    parser_setup.add_argument('--only-links', action='store_true')
    parser_setup.set_defaults(func=_setup)

    # Submit action
    parser_submit = subparsers.add_parser(
        'submit',
        help="Submits a job to a remote machine")
    add_destination_option(parser_submit)
    add_runtime_option(parser_submit)
    parser_submit.add_argument('--id', action='store',
                               help="Identifier for the new job")
    parser_submit.add_argument('--script', action='store',
                               help="Relative name of the script in the "
                                    "directory")
    parser_submit.add_argument('directory', action='store',
                               help="Job directory to upload")
    parser_submit.set_defaults(func=_submit)

    # Status action
    parser_status = subparsers.add_parser(
        'status',
        help="Gets the status of a job")
    add_destination_option(parser_status)
    parser_status.add_argument('--id', action='store',
                               help="Identifier of the running job")
    parser_status.set_defaults(func=_status)

    # Download action
    parser_download = subparsers.add_parser(
        'download',
        help="Downloads files from finished job")
    add_destination_option(parser_download)
    parser_download.add_argument('--id', action='store',
                                 help="Identifier of the job")
    parser_download.add_argument('files', action='store',
                                 nargs=argparse.ONE_OR_MORE,
                                 help="Files to download")
    parser_download.set_defaults(func=_download)

    # Kill action
    parser_kill = subparsers.add_parser(
        'kill',
        help="Kills a running job")
    add_destination_option(parser_kill)
    parser_kill.add_argument('--id', action='store',
                             help="Identifier of the running job")
    parser_kill.set_defaults(func=_kill)

    # Delete action
    parser_delete = subparsers.add_parser(
        'delete',
        help="Deletes a finished job")
    add_destination_option(parser_delete)
    parser_delete.add_argument('--id', action='store',
                               help="Identifier of the finished job")
    parser_delete.set_defaults(func=_delete)

    # List action
    parser_list = subparsers.add_parser(
        'list',
        help="Lists remote jobs")
    add_destination_option(parser_list)
    parser_list.set_defaults(func=_list)

    args = parser.parse_args()
    setup_logging(args.verbosity)

    try:
        args.func(args)
    except Error as e:
        # No need to show a traceback here, this is not an internal error
        logger.critical(e)
        sys.exit(1)
    sys.exit(0)