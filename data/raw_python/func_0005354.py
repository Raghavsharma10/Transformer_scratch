def changelog_cli(ctx):
    # type: () -> None
    """ Generate changelog from commit messages. """
    if ctx.invoked_subcommand:
        return

    from peltak.core import shell
    from . import logic
    shell.cprint(logic.changelog())