def init_db(self, app, entry_point_group='invenio_db.models', **kwargs):
        """Initialize Flask-SQLAlchemy extension."""
        # Setup SQLAlchemy
        app.config.setdefault(
            'SQLALCHEMY_DATABASE_URI',
            'sqlite:///' + os.path.join(app.instance_path, app.name + '.db')
        )
        app.config.setdefault('SQLALCHEMY_ECHO', False)

        # Initialize Flask-SQLAlchemy extension.
        database = kwargs.get('db', db)
        database.init_app(app)

        # Initialize versioning support.
        self.init_versioning(app, database, kwargs.get('versioning_manager'))

        # Initialize model bases
        if entry_point_group:
            for base_entry in pkg_resources.iter_entry_points(
                    entry_point_group):
                base_entry.load()

        # All models should be loaded by now.
        sa.orm.configure_mappers()
        # Ensure that versioning classes have been built.
        if app.config['DB_VERSIONING']:
            manager = self.versioning_manager
            if manager.pending_classes:
                if not versioning_models_registered(manager, database.Model):
                    manager.builder.configure_versioned_classes()
            elif 'transaction' not in database.metadata.tables:
                manager.declarative_base = database.Model
                manager.create_transaction_model()
                manager.plugins.after_build_tx_class(manager)