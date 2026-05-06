def get_db_args_env(self):
        """
        Get a dictionary of database connection parameters, and create an
        environment for running postgres commands.
        Falls back to omego defaults.
        """
        db = {
            'name': self.args.dbname,
            'host': self.args.dbhost,
            'user': self.args.dbuser,
            'pass': self.args.dbpass
            }

        if not self.args.no_db_config:
            try:
                c = self.external.get_config(force=True)
            except Exception as e:
                log.warn('config.xml not found: %s', e)
                c = {}

            for k in db:
                try:
                    db[k] = c['omero.db.%s' % k]
                except KeyError:
                    log.info(
                        'Failed to lookup parameter omero.db.%s, using %s',
                        k, db[k])

        if not db['name']:
            raise Exception('Database name required')

        env = os.environ.copy()
        env['PGPASSWORD'] = db['pass']
        return db, env