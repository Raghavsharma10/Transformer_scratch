def create_parser() -> FileAwareParser:
    """
    Create a command line parser
    :return: parser
    """
    parser = FileAwareParser(description="Clear data from FHIR observation fact table", prog="removefacts",
                             use_defaults=False)
    parser.add_argument("-ss", "--sourcesystem", metavar="SOURCE SYSTEM CODE", help="Sourcesystem code")
    parser.add_argument("-u", "--uploadid", metavar="UPLOAD IDENTIFIER",
                        help="Upload identifer -- uniquely identifies this batch", type=int,
                        nargs='*')
    add_connection_args(parser, strong_config_file=False)
    parser.add_argument("-p", "--testprefix", metavar="SS PREFIX",
                        help=f"Sourcesystem_cd prefix for test suite functions (Default: {default_test_prefix}")
    parser.add_argument("--testlist", help="List leftover test suite entries", action="store_true")
    parser.add_argument("--removetestlist", help="Remove leftover test suite entries", action="store_true")
    return parser