def cmd_rollback(self, name):
        """Rollback migrations."""
        from peewee_migrate.router import Router, LOGGER

        LOGGER.setLevel('INFO')
        LOGGER.propagate = 0

        router = Router(self.database,
                        migrate_dir=self.app.config['PEEWEE_MIGRATE_DIR'],
                        migrate_table=self.app.config['PEEWEE_MIGRATE_TABLE'])

        router.rollback(name)