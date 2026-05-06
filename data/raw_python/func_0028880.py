def _get_pidfile_path(self):
        """Return the normalized path for the pidfile, raising an
        exception if it can not written to.

        :return: str
        :raises: ValueError
        :raises: OSError

        """
        if self.config.daemon.pidfile:
            pidfile = path.abspath(self.config.daemon.pidfile)
            if not os.access(path.dirname(pidfile), os.W_OK):
                raise ValueError('Cannot write to specified pid file path'
                                 ' %s' % pidfile)
            return pidfile
        app = sys.argv[0].split('/')[-1]
        for pidfile in ['%s/pids/%s.pid' % (os.getcwd(), app),
                        '/var/run/%s.pid' % app,
                        '/var/run/%s/%s.pid' % (app, app),
                        '/var/tmp/%s.pid' % app,
                        '/tmp/%s.pid' % app,
                        '%s.pid' % app]:
            if os.access(path.dirname(pidfile), os.W_OK):
                return pidfile
        raise OSError('Could not find an appropriate place for a pid file')