def _commands(ctx):
    """Prints a list of commands for shell completion hooks."""
    ctx = ctx.parent
    ctx.show_hidden_subcommands = False
    main = ctx.command

    for subcommand in main.list_commands(ctx):
        cmd = main.get_command(ctx, subcommand)
        if cmd is None:
            continue
        help = cmd.short_help or ""
        click.echo("{}:{}".format(subcommand, help))