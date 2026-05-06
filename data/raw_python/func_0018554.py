def dump(self):
        """
        Dump the database using the postgres custom format
        """
        dumpfile = self.args.dumpfile
        if not dumpfile:
            db, env = self.get_db_args_env()
            dumpfile = fileutils.timestamp_filename(
                'omero-database-%s' % db['name'], 'pgdump')

        log.info('Dumping database to %s', dumpfile)
        if not self.args.dry_run:
            self.pgdump('-Fc', '-f', dumpfile)