def create_migration_template(name):
    """ Creates migration file. Returns created file name.
    Args:
        name (str): name of the migration.

    Returns:
        str: name of the migration file.
    """
    assert name, 'Name of the migration can not be empty.'
    from . import migrations

    # Find next number
    #
    package = migrations
    prefix = package.__name__ + '.'
    all_versions = []
    for importer, modname, ispkg in pkgutil.iter_modules(package.__path__, prefix):
        version = int(modname.split('.')[-1].split('_')[0])
        all_versions.append(version)

    next_number = max(all_versions) + 1

    # Generate next migration name
    #
    next_migration_name = '{}_{}.py'.format(next_number, name)
    migration_fullname = os.path.join(package.__path__[0], next_migration_name)

    # Write next migration file content.
    #
    with open(migration_fullname, 'w') as f:
        f.write(MIGRATION_TEMPLATE)
    return migration_fullname