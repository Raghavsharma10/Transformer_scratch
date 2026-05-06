def import_app(files, category, overwrite, id, name):
    """ Upload application from file.

    By default, file name will be used as application name, with "-vXX.YYY" suffix stripped.
    Application is looked up by one of these classifiers, in order of priority:
    app-id, app-name, filename.

    If app-id is provided, looks up existing application and updates its manifest.
    If app-id is NOT specified, looks up by name, or creates new application.

    """
    platform = _get_platform()
    org = platform.get_organization(QUBELL["organization"])
    if category:
        category = org.categories[category]
    regex = re.compile(r"^(.*?)(-v(\d+)|)\.[^.]+$")
    if (id or name) and len(files) > 1:
        raise Exception("--id and --name are supported only for single-file mode")

    for filename in files:
        click.echo("Importing " + filename, nl=False)
        if not name:
            match = regex.match(basename(filename))
            if not match:
                click.echo(_color("RED", "FAIL") + " unknown filename format")
                break
            name = regex.match(basename(filename)).group(1)
        click.echo(" => ", nl=False)
        app = None
        try:
            app = org.get_application(id=id, name=name)
            if app and not overwrite:
                click.echo("%s %s already exists %s" % (
                    app.id, _color("BLUE", app and app.name or name), _color("RED", "FAIL")))
                break
        except NotFoundError:
            if id:
                click.echo("%s %s not found %s" % (
                    id or "", _color("BLUE", app and app.name or name), _color("RED", "FAIL")))
                break
        click.echo(_color("BLUE", app and app.name or name) + " ", nl=False)
        try:
            with file(filename, "r") as f:
                if app:
                    app.update(name=app.name,
                               category=category and category.id or app.category,
                               manifest=Manifest(content=f.read()))
                else:
                    app = org.application(id=id, name=name, manifest=Manifest(content=f.read()))
                    if category:
                        app.update(category=category.id)
            click.echo(app.id + _color("GREEN", " OK"))
        except IOError as e:
            click.echo(_color("RED", " FAIL") + " " + e.message)
            break