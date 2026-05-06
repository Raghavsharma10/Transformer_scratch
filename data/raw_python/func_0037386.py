def psql_string(self, sql):
        """
        Evaluate the sql file (possibly multiple statements) using psql.
        """
        argv = [
            PostgresFinder.find_root() / 'psql',
            '--quiet',
            '-U', self.user,
            '-h', self.host,
            '-p', self.port,
            '-f', '-',
            self.db_name,
        ]
        popen = subprocess.Popen(argv, stdin=subprocess.PIPE)
        popen.communicate(input=sql.encode('utf-8'))
        if popen.returncode != 0:
            raise subprocess.CalledProcessError(popen.returncode, argv)