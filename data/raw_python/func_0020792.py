def list():
    """ List lambder functions """
    functions = lambder.list_functions()
    output = json.dumps(
        functions,
        sort_keys=True,
        indent=4,
        separators=(',', ':')
    )
    click.echo(output)