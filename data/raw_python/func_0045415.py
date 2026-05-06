def rcfile(appname, args={}, strip_dashes=True, module_name=None):
    """
        Read environment variables and config files and return them merged with predefined list of arguments.

        Arguments:
            appname - application name, used for config files and environemnt variable names.
            args - arguments from command line (optparse, docopt, etc).
            strip_dashes - strip dashes prefixing key names from args dict.

        Returns:
            dict containing the merged variables of environment variables, config files and args.

        Environment variables are read if they start with appname in uppercase with underscore, for example:

            TEST_VAR=1

        Config files compatible with ConfigParser are read and the section name appname is read, example:

            [appname]
            var=1

        Files are read from: /etc/appname/config, /etc/appfilerc, ~/.config/appname/config, ~/.config/appname,
            ~/.appname/config, ~/.appnamerc, .appnamerc, file provided by config variable in args.

        Example usage with docopt:

            args = rcfile(__name__, docopt(__doc__, version=__version__))
    """
    if strip_dashes:
        for k in args.keys():
            args[k.lstrip('-')] = args.pop(k)

    environ = get_environment(appname)

    if not module_name:
        module_name = appname

    config = get_config(appname, module_name, args.get('config', ''))

    return merge(merge(args, config), environ)