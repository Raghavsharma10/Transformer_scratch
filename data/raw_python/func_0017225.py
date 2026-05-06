def init_versioning(self, app, database, versioning_manager=None):
        """Initialize the versioning support using SQLAlchemy-Continuum."""
        try:
            pkg_resources.get_distribution('sqlalchemy_continuum')
        except pkg_resources.DistributionNotFound:  # pragma: no cover
            default_versioning = False
        else:
            default_versioning = True

        app.config.setdefault('DB_VERSIONING', default_versioning)

        if not app.config['DB_VERSIONING']:
            return

        if not default_versioning:  # pragma: no cover
            raise RuntimeError(
                'Please install extra versioning support first by running '
                'pip install invenio-db[versioning].'
            )

        # Now we can import SQLAlchemy-Continuum.
        from sqlalchemy_continuum import make_versioned
        from sqlalchemy_continuum import versioning_manager as default_vm
        from sqlalchemy_continuum.plugins import FlaskPlugin

        # Try to guess user model class:
        if 'DB_VERSIONING_USER_MODEL' not in app.config:  # pragma: no cover
            try:
                pkg_resources.get_distribution('invenio_accounts')
            except pkg_resources.DistributionNotFound:
                user_cls = None
            else:
                user_cls = 'User'
        else:
            user_cls = app.config.get('DB_VERSIONING_USER_MODEL')

        plugins = [FlaskPlugin()] if user_cls else []

        # Call make_versioned() before your models are defined.
        self.versioning_manager = versioning_manager or default_vm
        make_versioned(
            user_cls=user_cls,
            manager=self.versioning_manager,
            plugins=plugins,
        )

        # Register models that have been loaded beforehand.
        builder = self.versioning_manager.builder

        for tbl in database.metadata.tables.values():
            builder.instrument_versioned_classes(
                database.mapper, get_class_by_table(database.Model, tbl)
            )