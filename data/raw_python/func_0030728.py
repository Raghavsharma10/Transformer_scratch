def _get_all_migrations():
    """ Returns sorted list of all migrations.

    Returns:
        list of (int, str) tuples: first elem of the tuple is migration number, second if module name.

    """
    from . import migrations

    package = migrations
    prefix = package.__name__ + '.'
    all_migrations = []
    for importer, modname, ispkg in pkgutil.iter_modules(package.__path__, prefix):
        version = int(modname.split('.')[-1].split('_')[0])
        all_migrations.append((version, modname))

    all_migrations = sorted(all_migrations, key=lambda x: x[0])
    return all_migrations