def _autocomplete(ctx, shell):
    """Print the shell autocompletion code."""
    if not shell:
        shell = os.environ.get("SHELL", "")
        shell = os.path.basename(shell).lower()
    if not shell:
        click.secho(
            "Your shell could not be detected, please pass its name "
            "as the argument.",
            fg="red",
        )
        ctx.exit(-1)

    base = os.path.abspath(os.path.dirname(__file__))
    autocomplete = os.path.join(base, "autocomplete", "{}.sh".format(shell))

    if not os.path.exists(autocomplete):
        click.secho(
            "Autocompletion for your shell ({}) is currently not "
            "supported.",
            fg="red",
        )
        ctx.exit(-1)

    with open(autocomplete) as fh:
        click.echo(fh.read())