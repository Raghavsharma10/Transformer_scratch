def start(self):
        """Important:

            Do not extend this method, rather redefine Controller.run

        """
        for signum in [signal.SIGHUP, signal.SIGTERM,
                       signal.SIGUSR1, signal.SIGUSR2]:
            signal.signal(signum, self._on_signal)
        self.run()