def remove_facts(argv: List[str]) -> bool:
    """
    Convert a set of FHIR resources into their corresponding i2b2 counterparts.
    
    :param argv: Command line arguments.  See: create_parser for details
    :return:
    """
    parser = create_parser()
    local_opts = parser.parse_args(argv)                        # Pull everything from the actual command line
    if not (local_opts.uploadid or local_opts.sourcesystem or local_opts.testlist or local_opts.removetestlist):
        parser.error("Option must be one of: -ss, -u, --testlist, --removetestlist")

    if (local_opts.testlist or local_opts.removetestlist) and (local_opts.uploadid or local_opts.sourcesystem):
        parser.error("Cannot combine -ss or -u option with testlist options.  Use -p to specify ss prefix")

    opts, _ = parser.parse_known_args(parser.decode_file_args(argv))     # Include the options file
    if opts is None:
        return False
    opts.uploadid = local_opts.uploadid
    opts.sourcesystem = local_opts.sourcesystem

    process_parsed_args(opts, parser.error)           # Update CRC and Meta table connection information

    if opts.uploadid:
        for uploadid in opts.uploadid:
            print("---> Removing entries for id {}".format(uploadid))
            clear_i2b2_tables(I2B2Tables(opts), uploadid)
    if opts.sourcesystem:
        print("---> Removing entries for sourcesystem_cd {}".format(opts.sourcesystem))
        clear_i2b2_sourcesystems(I2B2Tables(opts), opts.sourcesystem)
    if opts.testlist:
        opts.testprefix = opts.testprefix if (opts and opts.testprefix) else default_test_prefix
        print(f"---> Listing orphan test elements for sourcesystem_cd starting with {opts.testprefix}")
        list_test_artifacts(opts)
    if opts.removetestlist:
        opts.testprefix = opts.testprefix if (opts and opts.testprefix) else default_test_prefix
        print(f"---> Removing orphan test elements for sourcesystem_cd starting with {opts.testprefix}")
        remove_test_artifacts(opts)
    return True