def start(self, block=False):
        """
        Start the daemon process.

        The daemon process is started in the background and the calling
        process returns.

        Once the daemon process is initialized it calls the
        :py:meth:`run` method.

        If ``block`` is true then the call blocks until the daemon
        process has started. ``block`` can either be ``True`` (in which
        case it blocks indefinitely) or a timeout in seconds.

        The return value is ``True`` if the daemon process has been
        started and ``False`` otherwise.

        .. versionadded:: 0.3
            The ``block`` parameter
        """
        pid = self.get_pid()
        if pid:
            raise ValueError('Daemon is already running at PID %d.' % pid)

        # The default is to place the PID file into ``/var/run``. This
        # requires root privileges. Since not having these is a common
        # problem we check a priori whether we can create the lock file.
        try:
            self.pid_file.acquire()
        finally:
            self.pid_file.release()

        # Clear previously received SIGTERMs. This must be done before
        # the calling process returns so that the calling process can
        # call ``stop`` directly after ``start`` returns without the
        # signal being lost.
        self.clear_signal(signal.SIGTERM)

        if _detach_process():
            # Calling process returns
            return _block(lambda: self.is_running(), block)
        # Daemon process continues here
        self._debug('Daemon has detached')

        def on_signal(s, frame):
            self._debug('Received signal {}'.format(s))
            self._signal_events[int(s)].set()

        def runner():
            try:
                # We acquire the PID as late as possible, since its
                # existence is used to verify whether the service
                # is running.
                self.pid_file.acquire()
                self._debug('PID file has been acquired')
                self._debug('Calling `run`')
                self.run()
                self._debug('`run` returned without exception')
            except Exception as e:
                self.logger.exception(e)
            except SystemExit:
                self._debug('`run` called `sys.exit`')
            try:
                self.pid_file.release()
                self._debug('PID file has been released')
            except Exception as e:
                self.logger.exception(e)
            os._exit(os.EX_OK)  # FIXME: This seems redundant

        try:
            setproctitle.setproctitle(self.name)
            self._debug('Process title has been set')
            files_preserve = (self.files_preserve +
                              self._get_logger_file_handles())
            signal_map = {s: on_signal for s in self._signal_events}
            signal_map.update({
                    signal.SIGTTIN: None,
                    signal.SIGTTOU: None,
                    signal.SIGTSTP: None,
            })
            with DaemonContext(
                    detach_process=False,
                    signal_map=signal_map,
                    files_preserve=files_preserve):
                self._debug('Daemon context has been established')

                # Python's signal handling mechanism only forwards signals to
                # the main thread and only when that thread is doing something
                # (e.g. not when it's waiting for a lock, etc.). If we use the
                # main thread for the ``run`` method this means that we cannot
                # use the synchronization devices from ``threading`` for
                # communicating the reception of SIGTERM to ``run``. Hence we
                # use  a separate thread for ``run`` and make sure that the
                # main loop receives signals. See
                # https://bugs.python.org/issue1167930
                thread = threading.Thread(target=runner)
                thread.start()
                while thread.is_alive():
                    time.sleep(1)
        except Exception as e:
            self.logger.exception(e)

        # We need to shutdown the daemon process at this point, because
        # otherwise it will continue executing from after the original
        # call to ``start``.
        os._exit(os.EX_OK)