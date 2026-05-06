def command_stop(self):
        '''
        Stop a server::

            ./manage.py flup:stop
        '''
        if self.pidfile:
            if not os.path.exists(self.pidfile):
                sys.exit("Pidfile {!r} doesn't exist".format(self.pidfile))
            with open(self.pidfile) as pidfile:
                pid = int(pidfile.read())
            for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGKILL]:
                try:
                    if terminate(pid, sig, 3):
                        os.remove(self.pidfile)
                        # NOTE: we are not performing sys.exit here,
                        # otherwise restart command will not work
                        return
                except OSError as exc:
                    if exc.errno != errno.ESRCH:
                        raise
                    elif sig == signal.SIGINT:
                        sys.exit('Not running')
        sys.exit('No pidfile provided')