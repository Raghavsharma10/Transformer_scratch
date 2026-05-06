def add_connection_args(parser: FileAwareParser, strong_config_file: bool=True) -> FileAwareParser:
    """
    Add the database connection arguments to the supplied parser

    :param parser: parser to add arguments to
    :param strong_config_file: If True, force --conf to be processed.  This is strictly a test for programming errors,
      and has to be skipped due to removefacts function.
    :return: parser
    """
    # TODO: Decide what to do with this
    parser.add_file_argument("--conf", metavar="CONFIG FILE", help="Configuration file",
                             action=ConfigFile if strong_config_file else None)

    parser.add_argument("-db", "--dburl", help="Default database URL",
                        default=Default_DB_Connection)
    parser.add_argument("--user", help="Default user name",
                        default=Default_User)
    parser.add_argument("--password", help="Default password",
                        default=Default_Password)
    parser.add_argument("--crcdb", help="CRC database URL. (default: dburl)")
    parser.add_argument("--crcuser", help="User name for CRC database. (default: user)")
    parser.add_argument("--crcpassword", help="Password for CRC database. (default: password)")
    parser.add_argument("--ontodb", help="Ontology database URL.  (default: dburl)")
    parser.add_argument("--ontouser", help="User name for ontology database. (default: user)")
    parser.add_argument("--ontopassword", help="Password for ontology database. (default: password)")
    parser.add_argument("--onttable", metavar="ONTOLOGY TABLE NAME",
                        help="Ontology table name (default: {})".format(DEFAULT_ONTOLOGY_TABLE),
                        default=DEFAULT_ONTOLOGY_TABLE)
    return parser