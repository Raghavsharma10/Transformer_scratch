def generate_conf_file(argv: List[str]) -> bool:
    """
    Convert a set of FHIR resources into their corresponding i2b2 counterparts.

    :param argv: Command line arguments.  See: create_parser for details
    :return:
    """
    parser = ArgumentParser(description="Generate SQL db_conf file template")
    parser.add_argument("-f", "--configfile", help="File name to generate (Default: db_conf)", metavar="Config File",
                        default="db_conf")
    opts = parser.parse_args(argv)
    if os.path.exists(opts.configfile):
        print(f"{opts.configfile} already exists!")
        return False
    with open(opts.configfile, 'w') as f:
        f.write(conf_template)
    print(f"{opts.configfile} generated")
    return True