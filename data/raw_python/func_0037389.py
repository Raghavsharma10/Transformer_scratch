def super_psql(self, args):
        """Just like .psql(), except that we connect as the database superuser
        (and we connect to the superuser's database, not the user's database).
        """
        argv = [
            PostgresFinder.find_root() / 'psql',
            '--quiet',
            '-U', self.superuser,
            '-h', self.host,
            '-p', self.port,
        ] + args
        subprocess.check_call(argv)