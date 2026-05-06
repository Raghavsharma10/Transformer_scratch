def cmd_create(self, name, auto=False):
        """Create a new migration."""

        LOGGER.setLevel('INFO')
        LOGGER.propagate = 0

        router = Router(self.database,
                        migrate_dir=self.app.config['PEEWEE_MIGRATE_DIR'],
                        migrate_table=self.app.config['PEEWEE_MIGRATE_TABLE'])

        if auto:
            auto = self.models

        router.create(name, auto=auto)