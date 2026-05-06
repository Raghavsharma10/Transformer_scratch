def setup(self):
        """
        Initialize the crochet library.

        This starts the reactor in a thread, and connect's Twisted's logs to
        Python's standard library logging module.

        This must be called at least once before the library can be used, and
        can be called multiple times.
        """
        if self._started:
            return
        self._common_setup()
        if platform.type == "posix":
            self._reactor.callFromThread(self._startReapingProcesses)
        if self._startLoggingWithObserver:
            observer = ThreadLogObserver(PythonLoggingObserver().emit)

            def start():
                # Twisted is going to override warnings.showwarning; let's
                # make sure that has no effect:
                from twisted.python import log
                original = log.showwarning
                log.showwarning = warnings.showwarning
                self._startLoggingWithObserver(observer, False)
                log.showwarning = original

            self._reactor.callFromThread(start)

            # We only want to stop the logging thread once the reactor has
            # shut down:
            self._reactor.addSystemEventTrigger(
                "after", "shutdown", observer.stop)
        t = threading.Thread(
            target=lambda: self._reactor.run(installSignalHandlers=False),
            name="CrochetReactor")
        t.start()
        self._atexit_register(self._reactor.callFromThread, self._reactor.stop)
        self._atexit_register(_store.log_errors)
        if self._watchdog_thread is not None:
            self._watchdog_thread.start()