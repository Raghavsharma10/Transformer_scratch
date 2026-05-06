def main():
    """Primary entry point for all AstroCats catalogs.

    From this entry point, all internal catalogs can be accessed and their
    public methods executed (for example: import scripts).

    """
    from datetime import datetime

    # Initialize Command-Line and User-Config Settings, Log
    # -----------------------------------------------------

    beg_time = datetime.now()
    # Process command-line arguments to determine action
    # If no subcommand (e.g. 'import') is given, returns 'None' --> exit
    args, sub_clargs = load_command_line_args()
    if args is None:
        return

    # Create a logging object
    log = load_log(args)

    # Run configuration/setup interactive script
    if args.command == 'setup':
        setup_user_config(log)
        return

    # Make sure configuration file exists, or that's what we're doing
    # (with the 'setup' subcommand)
    if not os.path.isfile(_CONFIG_PATH):
        raise RuntimeError("'{}' does not exist.  "
                           "Run `astrocats setup` to configure."
                           "".format(_CONFIG_PATH))

    git_vers = get_git()
    title_str = "Astrocats, version: {}, SHA: {}".format(__version__, git_vers)
    log.warning("\n\n{}\n{}\n{}\n".format(title_str, '=' * len(title_str),
                                          beg_time.ctime()))

    # Load the user settings from the home directory
    args = load_user_config(args, log)

    # Choose Catalog and Operation(s) to perform
    # ------------------------------------------
    mod_name = args.command
    log.debug("Importing specified module: '{}'".format(mod_name))
    # Try to import the specified module
    try:
        mod = importlib.import_module('.' + mod_name, package='astrocats')
    except Exception as err:
        log.error("Import of specified module '{}' failed.".format(mod_name))
        log_raise(log, str(err), type(err))

    # Run the `main.main` method of the specified module
    log.debug("Running `main.main()`")
    mod.main.main(args, sub_clargs, log)

    end_time = datetime.now()
    log.warning("\nAll complete at {}, After {}".format(end_time, end_time -
                                                        beg_time))
    return