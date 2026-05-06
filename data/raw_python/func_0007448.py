def version():
    """Show pbs version."""
    try:
        import pkg_resources
        click.echo(pkg_resources.get_distribution('pybossa-pbs').version)
    except ImportError:
        click.echo("pybossa-pbs package not found!")