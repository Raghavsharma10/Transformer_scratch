def locate(command, on):
    """Locate the command's man page."""
    location = find_page_location(command, on)
    click.echo(location)