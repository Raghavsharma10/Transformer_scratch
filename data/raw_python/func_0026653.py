def cli(ctx, instance, quiet, verbose, log_level, dbhost, dbname):
    """Isomer Management Tool

    This tool supports various operations to manage isomer instances.

    Most of the commands are grouped. To obtain more information about the
    groups' available sub commands/groups, try

    iso [group]

    To display details of a command or its sub groups, try

    iso [group] [subgroup] [..] [command] --help

    To get a map of all available commands, try

    iso cmdmap
    """

    ctx.obj['instance'] = instance

    if dbname == db_default and instance != 'default':
        dbname = instance

    ctx.obj['quiet'] = quiet
    ctx.obj['verbose'] = verbose
    verbosity['console'] = log_level
    verbosity['global'] = log_level

    ctx.obj['dbhost'] = dbhost
    ctx.obj['dbname'] = dbname