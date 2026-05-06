def cmd_merge(self):
        """Merge migrations."""
        from peewee_migrate.router import Router, LOGGER

        LOGGER.setLevel('DEBUG')
        LOGGER.propagate = 0

        router = Router(self.database,
                        migrate_dir=self.app.config['PEEWEE_MIGRATE_DIR'],
                        migrate_table=self.app.config['PEEWEE_MIGRATE_TABLE'])

        router.merge()