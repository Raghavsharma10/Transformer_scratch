def start(self):
        """Launch this postgres server.  If it's already running, do nothing.

        If the backing storage directory isn't configured, raise
        NotInitializedError.

        This method is optional.  If you're running in an environment
        where the DBMS is provided as part of the basic infrastructure,
        you probably want to skip this step!
        """
        log.info('Starting PostgreSQL at %s:%s', self.host, self.port)
        if not self.base_pathname:
            tmpl = ('Invalid base_pathname: %r.  Did you forget to call '
                    '.initdb()?')
            raise NotInitializedError(tmpl % self.base_pathname)

        conf_file = os.path.join(self.base_pathname, 'postgresql.conf')
        if not os.path.exists(conf_file):
            tmpl = 'No config file at: %r.  Did you forget to call .initdb()?'
            raise NotInitializedError(tmpl % self.base_pathname)

        if not self.is_running():
            version = self.get_version()
            if version and version >= (9, 3):
                socketop = 'unix_socket_directories'
            else:
                socketop = 'unix_socket_directory'
            postgres_options = [
                # When running not as root, postgres might try to put files
                #  where they're not writable (see
                #  https://paste.yougov.net/YKdgi). So set the socket_dir.
                '-c', '{}={}'.format(socketop, self.base_pathname),
                '-h', self.host,
                '-i',  # enable TCP/IP connections
                '-p', self.port,
            ]
            subprocess.check_call([
                PostgresFinder.find_root() / 'pg_ctl',
                'start',
                '-D', self.base_pathname,
                '-l', os.path.join(self.base_pathname, 'postgresql.log'),
                '-o', subprocess.list2cmdline(postgres_options),
            ])

        # Postgres may launch, then abort if it's unhappy with some parameter.
        # This post-launch test helps us decide.
        if not self.is_running():
            tmpl = ('%s aborted immediately after launch, check '
                    'postgresql.log in storage dir')
            raise RuntimeError(tmpl % self)