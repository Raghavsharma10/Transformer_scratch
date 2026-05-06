def command_serve(self, host='', port='8000', level='debug'):
        '''
        Run development server with automated reload on code change::

            ./manage.py app:serve [host] [port] [level]
        '''
        logging.basicConfig(level=getattr(logging, level.upper()), format=self.format)
        if self.bootstrap:
            logger.info('Bootstraping...')
            self.bootstrap()
        try:
            server_thread = DevServerThread(host, port, self.app)
            server_thread.start()

            wait_for_code_change(extra_files=self.extra_files)
            server_thread.running = False
            server_thread.join()
            logger.info('Reloading...')
            flush_fds()
            pid = os.fork()
            # We need to fork before `execvp` to perform code reload
            # correctly, because we need to complete python destructors and
            # `atexit`.
            # This will save us from problems of incorrect exit, such as:
            # - unsaved data in data storage, which does not write data
            # on hard drive immediatly
            # - code, that can't be measured with coverage tool, because it uses
            # `atexit` handler to save coverage data
            # NOTE: we using untipical fork-exec scheme with replacing
            # the parent process(not the child) to preserve PID of proccess
            # we use `pragma: no cover` here, because parent process cannot be
            # measured with coverage since it is ends with `execvp`
            if pid: # pragma: no cover
                os.closerange(3, MAXFD)
                os.waitpid(pid, 0)
                # reloading the code in parent process
                os.execvp(sys.executable, [sys.executable] + sys.argv)
            else:
                # we closing our recources, including file descriptors
                # and performing `atexit`.
                sys.exit()
        except KeyboardInterrupt:
            logger.info('Stoping dev-server...')
            server_thread.running = False
            server_thread.join()
            sys.exit()