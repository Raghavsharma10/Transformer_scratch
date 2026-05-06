def _arguments(ctx, command_name=None):
    """Prints a list of arguments for shell completion hooks.

    If a command name is given, returns the arguments for that subcommand.
    The command name has to refer to a command; aliases are not supported.
    """
    ctx = ctx.parent
    main = ctx.command
    if command_name:
        command = main.get_command(ctx, command_name)
        if not command:
            return
    else:
        command = main

    types = ["option", "argument"]
    all_params = sorted(
        command.get_params(ctx), key=lambda p: types.index(p.param_type_name)
    )

    def get_name(param):
        return max(param.opts, key=len)

    for param in all_params:
        if param.param_type_name == "option":
            option = get_name(param)
            same_dest = [
                get_name(p) for p in all_params if p.name == param.name
            ]
            if same_dest:
                option = "({})".format(" ".join(same_dest)) + option
            if param.help:
                option += "[{}]".format(param.help or "")
            if not param.is_flag:
                option += "=:( )"
            click.echo(option)
        elif param.param_type_name == "argument":
            option = get_name(param)
            click.echo(":{}".format(option))