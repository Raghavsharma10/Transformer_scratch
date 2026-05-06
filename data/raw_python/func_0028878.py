def _daemonize(self):
        """Fork into a background process and setup the process, copied in part
        from http://www.jejik.com/files/examples/daemon3x.py

        """
        LOGGER.info('Forking %s into the background', sys.argv[0])

        # Write the pidfile if current uid != final uid
        if os.getuid() != self.uid:
            fd = open(self.pidfile_path, 'w')
            os.fchmod(fd.fileno(), 0o644)
            os.fchown(fd.fileno(), self.uid, self.gid)
            fd.close()

        try:
            pid = os.fork()
            if pid > 0:
                    sys.exit(0)
        except OSError as error:
                raise OSError('Could not fork off parent: %s', error)

        # Set the user id
        if self.uid != os.getuid():
            os.setuid(self.uid)

        # Set the group id
        if self.gid != os.getgid():
            try:
                os.setgid(self.gid)
            except OSError as error:
                LOGGER.error('Could not set group: %s', error)

        # Decouple from parent environment
        os.chdir('/')
        os.setsid()
        os.umask(0o022)

        # Fork again
        try:
            pid = os.fork()
            if pid > 0:
                sys.exit(0)
        except OSError as error:
            raise OSError('Could not fork child: %s', error)

        # redirect standard file descriptors
        sys.stdout.flush()
        sys.stderr.flush()
        si = open(os.devnull, 'r')
        so = open(os.devnull, 'a+')
        se = open(os.devnull, 'a+')
        os.dup2(si.fileno(), sys.stdin.fileno())
        os.dup2(so.fileno(), sys.stdout.fileno())
        os.dup2(se.fileno(), sys.stderr.fileno())

        # Automatically call self._remove_pidfile when the app exits
        atexit.register(self._remove_pidfile)
        self._write_pidfile()