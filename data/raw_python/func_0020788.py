def list():
    """ List all events """
    entries = lambder.list_events()
    for e in entries:
        click.echo(str(e))