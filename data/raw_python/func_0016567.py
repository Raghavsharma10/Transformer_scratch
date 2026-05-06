def ls(system, user, local, include_missing):
    """List configuration files detected (and/or examined paths)."""

    # default action is to list *all* auto-detected files
    if not (system or user or local):
        system = user = local = True

    for path in get_configfile_paths(system=system, user=user, local=local,
                                     only_existing=not include_missing):
        click.echo(path)