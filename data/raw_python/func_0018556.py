def psql(self, *psqlargs):
        """
        Run a psql command
        """
        db, env = self.get_db_args_env()

        args = [
            '-v', 'ON_ERROR_STOP=on',
            '-d', db['name'],
            '-h', db['host'],
            '-U', db['user'],
            '-w', '-A', '-t'
            ] + list(psqlargs)
        stdout, stderr = External.run('psql', args, capturestd=True, env=env)
        if stderr:
            log.warn('stderr: %s', stderr)
        log.debug('stdout: %s', stdout)
        return stdout