def cmd_list(self):
        """List migrations."""
        from peewee_migrate.router import Router, LOGGER

        LOGGER.setLevel('DEBUG')
        LOGGER.propagate = 0

        router = Router(self.database,
                        migrate_dir=self.app.config['PEEWEE_MIGRATE_DIR'],
                        migrate_table=self.app.config['PEEWEE_MIGRATE_TABLE'])

        LOGGER.info('Migrations are done:')
        LOGGER.info('\n'.join(router.done))
        LOGGER.info('')
        LOGGER.info('Migrations are undone:')
        LOGGER.info('\n'.join(router.diff))