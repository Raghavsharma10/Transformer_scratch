def parse_args(cmd_args, is_script=False):
    """Parses a list of command line arguments into a ValidationOptions object.

    Args:
        cmd_args (list of str): The list of command line arguments to be parsed.
        is_script: Whether the arguments are intended for use in a stand-alone
            script or imported into another tool.

    Returns:
        Instance of ``ValidationOptions``

    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=NewlinesHelpFormatter,
        epilog=CODES_TABLE
    )

    # Input options
    if is_script:
        parser.add_argument(
            "files",
            metavar="FILES",
            nargs="*",
            default=sys.stdin,
            help="A whitespace separated list of STIX files or directories of "
                 "STIX files to validate. If none given, stdin will be used."
        )
    parser.add_argument(
        "-r",
        "--recursive",
        dest="recursive",
        action="store_true",
        default=True,
        help="Recursively descend into input directories."
    )
    parser.add_argument(
        "-s",
        "--schemas",
        dest="schema_dir",
        help="Custom schema directory. If provided, input will be validated "
             "against these schemas in addition to the STIX schemas bundled "
             "with this script."
    )
    parser.add_argument(
        "--version",
        dest="version",
        default=DEFAULT_VER,
        help="The version of the STIX specification to validate against (e.g. "
             "\"2.0\")."
    )

    # Output options
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        help="Print informational notes and more verbose error messages."
    )

    parser.add_argument(
        "-q",
        "--silent",
        dest="silent",
        action="store_true",
        default=False,
        help="Silence all output to stdout."
    )

    parser.add_argument(
        "-d",
        "--disable",
        "--ignore",
        dest="disabled",
        default="",
        help="A comma-separated list of recommended best practice checks to "
             "skip. By default, no checks are disabled. \n\n"
             "Example: --disable 202,210"
    )

    parser.add_argument(
        "-e",
        "--enable",
        "--select",
        dest="enabled",
        default="",
        help="A comma-separated list of recommended best practice checks to "
             "enable. If the --disable option is not used, no other checks "
             "will be run. By default, all checks are enabled.\n\n"
             "Example: --enable 218"
    )

    parser.add_argument(
        "--strict",
        dest="strict",
        action="store_true",
        default=False,
        help="Treat warnings as errors and fail validation if any are found."
    )

    parser.add_argument(
        "--strict-types",
        dest="strict_types",
        action="store_true",
        default=False,
        help="Ensure that no custom object types are used, only those defined"
             " in the STIX specification."
    )

    parser.add_argument(
        "--strict-properties",
        dest="strict_properties",
        action="store_true",
        default=False,
        help="Ensure that no custom properties are used, only those defined"
             " in the STIX specification."
    )

    parser.add_argument(
        "--no-cache",
        dest="no_cache",
        action="store_true",
        default=False,
        help="Disable the caching of external source values."
    )

    parser.add_argument(
        "--refresh-cache",
        dest="refresh_cache",
        action="store_true",
        default=False,
        help="Clears the cache of external source values, then "
             "during validation downloads them again."
    )

    parser.add_argument(
        "--clear-cache",
        dest="clear_cache",
        action="store_true",
        default=False,
        help="Clear the cache of external source values after validation."
    )

    parser.add_argument(
        "--enforce-refs",
        dest="enforce_refs",
        action="store_true",
        default=False,
        help="Ensures that all SDOs being referenced by SROs are contained "
             "within the same bundle."
    )

    args = parser.parse_args(cmd_args)

    if not is_script:
        args.files = ""
    if not args.version:
        args.version = DEFAULT_VER

    return ValidationOptions(args)