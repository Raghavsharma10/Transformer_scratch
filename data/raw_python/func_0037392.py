def initdb(self, quiet=True, locale='en_US.UTF-8'):
        """Bootstrap this DBMS from nothing.

        If you're running in an environment where the DBMS is provided as part
        of the basic infrastructure, you probably don't want to call this
        method!

        @param quiet: Should we operate quietly, emitting nothing if things go
        well?
        """
        # Defining base_pathname is deferred until this point because we don't
        # want to create a temp directory unless it's needed.  And now it is!
        if self.base_pathname in [None, '']:
            self.base_pathname = tempfile.mkdtemp()
        if not os.path.isdir(self.base_pathname):
            os.mkdir(self.base_pathname)
        stdout = DEV_NULL if quiet else None
        # The database superuser needs no password at this point(!).
        arguments = [
            '--auth=trust',
            '--username', self.superuser,
        ]
        if locale is not None:
            arguments.extend(('--locale', locale))
        cmd = [
            PostgresFinder.find_root() / 'initdb',
        ] + arguments + ['--pgdata', self.base_pathname]
        log.info('Initializing PostgreSQL with command: {}'.format(
            ' '.join(cmd)
        ))
        subprocess.check_call(cmd, stdout=stdout)