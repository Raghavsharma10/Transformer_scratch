def cli_run():
    """Run the daemon from a command line interface"""
    options = CLI.parse_args()
    run(options.CONFIGURATION, options.log_level, options.log_target, options.log_journal)