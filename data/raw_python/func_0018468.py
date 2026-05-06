def find(command, on):
    """Find the command usage."""
    output_lines = parse_man_page(command, on)
    click.echo(''.join(output_lines))