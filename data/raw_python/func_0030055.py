def parse_conf_args(argv):
    """Parse command line options into {section: (option, key)} which can be
    used for overlaying on top of config

    :param argv: list of argumets to be parsed
    :return: Dictionary in the following format: {section: (option, key)}"""

    args = {}
    for rarg in argv:
        if re.match("^--.*", rarg):
            arg = rarg.replace('--','', 1)

            fsplit = arg.split('=', 1)
            if len(fsplit) != 2:
                raise Exception(
                    "Command option '%s' not recognized." % rarg)

            rkey, value = fsplit
            ssplit = rkey.split('.', 1)
            if len(ssplit) != 2 or not ssplit[1]:
                raise Exception(
                    "Command option '%s' not recognized." % rarg)

            section, option = ssplit
            args[section] = (option, value)
        else:
            raise Exception(
                    "Command option '%s' not recognized." % rarg)

    return args