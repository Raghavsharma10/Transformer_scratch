def check_migrations_applied(app_configs, **kwargs):
    """
    A Django check to see if all migrations have been applied correctly.
    """
    from django.db.migrations.loader import MigrationLoader
    errors = []

    # Load migrations from disk/DB
    try:
        loader = MigrationLoader(connection, ignore_no_migrations=True)
    except (ImproperlyConfigured, ProgrammingError, OperationalError):
        msg = "Can't connect to database to check migrations"
        return [checks.Info(msg, id=health.INFO_CANT_CHECK_MIGRATIONS)]

    if app_configs:
        app_labels = [app.label for app in app_configs]
    else:
        app_labels = loader.migrated_apps

    for node, migration in loader.graph.nodes.items():
        if migration.app_label not in app_labels:
            continue
        if node not in loader.applied_migrations:
            msg = 'Unapplied migration {}'.format(migration)
            # NB: This *must* be a Warning, not an Error, because Errors
            # prevent migrations from being run.
            errors.append(checks.Warning(msg,
                                         id=health.WARNING_UNAPPLIED_MIGRATION))

    return errors