def start(self):
        """Daemonize if the process is not already running."""
        if self._is_already_running():
            LOGGER.error('Is already running')
            sys.exit(1)
        try:
            self._daemonize()
            self.controller.start()
        except Exception as error:
            sys.stderr.write('\nERROR: Startup of %s Failed\n.' %
                             sys.argv[0].split('/')[-1])
            exception_log = self._get_exception_log_path()
            if exception_log:
                with open(exception_log, 'a') as handle:
                    timestamp = datetime.datetime.now().isoformat()
                    handle.write('{:->80}\n'.format(' [START]'))
                    handle.write('%s Exception [%s]\n' % (sys.argv[0],
                                                          timestamp))
                    handle.write('{:->80}\n'.format(' [INFO]'))
                    handle.write('Interpreter: %s\n' % sys.executable)
                    handle.write('CLI arguments: %s\n' % ' '.join(sys.argv))
                    handle.write('Exception: %s\n' % error)
                    handle.write('Traceback:\n')
                    output = traceback.format_exception(*sys.exc_info())
                    _dev_null = [(handle.write(line),
                                 sys.stdout.write(line)) for line in output]
                    handle.write('{:->80}\n'.format(' [END]'))
                    handle.flush()
                sys.stderr.write('\nException log: %s\n\n' % exception_log)
            sys.exit(1)