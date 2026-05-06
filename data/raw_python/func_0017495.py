def cmd_migrate(self, name=None, fake=False):
        """Run migrations."""
        from peewee_migrate.router import Router, LOGGER

        LOGGER.setLevel('INFO')
        LOGGER.propagate = 0

        router = Router(self.database,
                        migrate_dir=self.app.config['PEEWEE_MIGRATE_DIR'],
                        migrate_table=self.app.config['PEEWEE_MIGRATE_TABLE'])

        migrations = router.run(name, fake=fake)
        if migrations:
            LOGGER.warn('Migrations are completed: %s' % ', '.join(migrations))