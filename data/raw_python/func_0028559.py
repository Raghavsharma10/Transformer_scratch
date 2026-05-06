def check_migrations_applied(migrate):
    """
    A built-in check to see if all migrations have been applied correctly.

    It's automatically added to the list of Dockerflow checks if a
    `flask_migrate.Migrate <https://flask-migrate.readthedocs.io/>`_ object
    is passed to the :class:`~dockerflow.flask.app.Dockerflow` class during
    instantiation, e.g.::

        from flask import Flask
        from flask_migrate import Migrate
        from flask_sqlalchemy import SQLAlchemy
        from dockerflow.flask import Dockerflow

        app = Flask(__name__)
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////tmp/test.db'
        db = SQLAlchemy(app)
        migrate = Migrate(app, db)

        dockerflow = Dockerflow(app, db=db, migrate=migrate)
    """
    errors = []

    from alembic.migration import MigrationContext
    from alembic.script import ScriptDirectory
    from sqlalchemy.exc import DBAPIError, SQLAlchemyError

    # pass in Migrate.directory here explicitly to be compatible with
    # older versions of Flask-Migrate that required the directory to be passed
    config = migrate.get_config(directory=migrate.directory)
    script = ScriptDirectory.from_config(config)

    try:
        with migrate.db.engine.connect() as connection:
            context = MigrationContext.configure(connection)
            db_heads = set(context.get_current_heads())
            script_heads = set(script.get_heads())
    except (DBAPIError, SQLAlchemyError) as e:
        msg = "Can't connect to database to check migrations: {!s}".format(e)
        return [Info(msg, id=health.INFO_CANT_CHECK_MIGRATIONS)]

    if db_heads != script_heads:
        msg = "Unapplied migrations found: {}".format(', '.join(script_heads))
        errors.append(Warning(msg, id=health.WARNING_UNAPPLIED_MIGRATION))
    return errors