def cli(config, server, api_key, all, credentials, project):
    """Create the cli command line."""
    # Check first for the pybossa.rc file to configure server and api-key
    home = expanduser("~")
    if os.path.isfile(os.path.join(home, '.pybossa.cfg')):
        config.parser.read(os.path.join(home, '.pybossa.cfg'))
        config.server = config.parser.get(credentials,'server')
        config.api_key = config.parser.get(credentials, 'apikey')
        try:
            config.all = config.parser.get(credentials, 'all')
        except ConfigParser.NoOptionError:
            config.all = None
    if server:
        config.server = server
    if api_key:
        config.api_key = api_key
    if all:
        config.all = all
    try:
        config.project = json.loads(project.read())
    except JSONDecodeError as e:
        click.secho("Error: invalid JSON format in project.json:", fg='red')
        if e.msg == 'Expecting value':
            e.msg += " (if string enclose it with double quotes)"
        click.echo("%s\n%s: line %s column %s" % (e.doc, e.msg, e.lineno, e.colno))
        raise click.Abort()
    try:
        project_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "short_name": {"type": "string"},
                "description": {"type": "string"}
            }
        }
        jsonschema.validate(config.project, project_schema)
    except jsonschema.exceptions.ValidationError as e:
        click.secho("Error: invalid type in project.json", fg='red')
        click.secho("'%s': %s" % (e.path[0], e.message), fg='yellow')
        click.echo("'%s' must be a %s" % (e.path[0], e.validator_value))
        raise click.Abort()

    config.pbclient = pbclient
    config.pbclient.set('endpoint', config.server)
    config.pbclient.set('api_key', config.api_key)