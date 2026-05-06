def init_app(self, app, database=None):
        """Initialize application."""

        # Register application
        if not app:
            raise RuntimeError('Invalid application.')
        self.app = app
        if not hasattr(app, 'extensions'):
            app.extensions = {}
        app.extensions['peewee'] = self

        app.config.setdefault('PEEWEE_CONNECTION_PARAMS', {})
        app.config.setdefault('PEEWEE_DATABASE_URI', 'sqlite:///peewee.sqlite')
        app.config.setdefault('PEEWEE_MANUAL', False)
        app.config.setdefault('PEEWEE_MIGRATE_DIR', 'migrations')
        app.config.setdefault('PEEWEE_MIGRATE_TABLE', 'migratehistory')
        app.config.setdefault('PEEWEE_MODELS_CLASS', Model)
        app.config.setdefault('PEEWEE_MODELS_IGNORE', [])
        app.config.setdefault('PEEWEE_MODELS_MODULE', '')
        app.config.setdefault('PEEWEE_READ_SLAVES', '')
        app.config.setdefault('PEEWEE_USE_READ_SLAVES', True)

        # Initialize database
        params = app.config['PEEWEE_CONNECTION_PARAMS']
        database = database or app.config.get('PEEWEE_DATABASE_URI')
        if not database:
            raise RuntimeError('Invalid database.')
        database = get_database(database, **params)

        slaves = app.config['PEEWEE_READ_SLAVES']
        if isinstance(slaves, string_types):
            slaves = slaves.split(',')
        self.slaves = [get_database(slave, **params) for slave in slaves if slave]

        self.database.initialize(database)
        if self.database.database == ':memory:':
            app.config['PEEWEE_MANUAL'] = True

        if not app.config['PEEWEE_MANUAL']:
            app.before_request(self.connect)
            app.teardown_request(self.close)