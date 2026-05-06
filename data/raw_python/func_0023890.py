def add_logging_parser(main_parser):
    "Build an argparse argument parser to parse the command line."

    main_parser.set_defaults(setup_logging=set_logging_level)

    verbosity_group = main_parser.add_mutually_exclusive_group(required=False)
    verbosity_group.add_argument(
        '--verbose',
        '-v',
        action='count',
        help='Output more verbose logging. Can be specified multiple times.')
    verbosity_group.add_argument(
        '--quiet',
        '-q',
        action='count',
        help='Output less information to the console during operation. Can be \
            specified multiple times.')

    main_parser.add_argument(
        '--silence-urllib3',
        action='store_true',
        help='Silence urllib3 warnings. See '
        'https://urllib3.readthedocs.org/en/latest/security.html for details.')

    return verbosity_group