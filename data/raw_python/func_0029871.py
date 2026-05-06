def command_start(self, daemonize=False):
        '''
        Start a server::

            ./manage.py flup:start [--daemonize]
        '''
        if daemonize:
            safe_makedirs(self.logfile, self.pidfile)
        flup_fastcgi(self.app, bind=self.bind, pidfile=self.pidfile,
                     logfile=self.logfile, daemonize=daemonize,
                     cwd=self.cwd, umask=self.umask, **self.fastcgi_params)