def pgdump(self, *pgdumpargs):
        """
        Run a pg_dump command
        """
        db, env = self.get_db_args_env()

        args = ['-d', db['name'], '-h', db['host'], '-U', db['user'], '-w'
                ] + list(pgdumpargs)
        stdout, stderr = External.run(
            'pg_dump', args, capturestd=True, env=env)
        if stderr:
            log.warn('stderr: %s', stderr)
        log.debug('stdout: %s', stdout)
        return stdout