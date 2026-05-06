def check_database_connected(db):
    """
    A built-in check to see if connecting to the configured default
    database backend succeeds.

    It's automatically added to the list of Dockerflow checks if a
    :class:`~flask_sqlalchemy.SQLAlchemy` object is passed
    to the :class:`~dockerflow.flask.app.Dockerflow` class during
    instantiation, e.g.::

        from flask import Flask
        from flask_sqlalchemy import SQLAlchemy
        from dockerflow.flask import Dockerflow

        app = Flask(__name__)
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////tmp/test.db'
        db = SQLAlchemy(app)

        dockerflow = Dockerflow(app, db=db)
    """
    from sqlalchemy.exc import DBAPIError, SQLAlchemyError

    errors = []
    try:
        with db.engine.connect() as connection:
            connection.execute('SELECT 1;')
    except DBAPIError as e:
        msg = 'DB-API error: {!s}'.format(e)
        errors.append(Error(msg, id=health.ERROR_DB_API_EXCEPTION))
    except SQLAlchemyError as e:
        msg = 'Database misconfigured: "{!s}"'.format(e)
        errors.append(Error(msg, id=health.ERROR_SQLALCHEMY_EXCEPTION))
    return errors