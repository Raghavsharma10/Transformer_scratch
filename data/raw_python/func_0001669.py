def _setup_helper():
    """Print the shell integration code."""
    base = os.path.abspath(os.path.dirname(__file__))
    helper = os.path.join(base, "helper.sh")
    with open(helper) as fh:
        click.echo(fh.read())