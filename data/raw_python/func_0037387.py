def psql(self, args):
        r"""Invoke psql, passing the given command-line arguments.

        Typical <args> values: ['-c', <sql_string>] or ['-f', <pathname>].

        Connection parameters are taken from self.  STDIN, STDOUT,
        and STDERR are inherited from the parent.

        WARNING: This method uses the psql(1) program, which ignores SQL
        errors
        by default.  That hides many real errors, making our software less
        reliable.  To overcome this flaw, add this line to the head of your
        SQL:
            "\set ON_ERROR_STOP TRUE"

        @return: None. Raises an exception upon error, but *ignores SQL
        errors* unless "\set ON_ERROR_STOP TRUE" is used.
        """
        argv = [
            PostgresFinder.find_root() / 'psql',
            '--quiet',
            '-U', self.user,
            '-h', self.host,
            '-p', self.port,
        ] + args + [self.db_name]
        subprocess.check_call(argv)