def start(self):
        """
        The main method that starts the service. This is blocking.

        """
        self._initial_setup()
        self.on_service_start()

        self.app = self.make_tornado_app()
        enable_pretty_logging()
        self.app.listen(self.port, address=self.host)

        self._start_periodic_tasks()
        # starts the event handlers
        self._initialize_event_handlers()
        self._start_event_handlers()

        try:
            self.io_loop.start()
        except RuntimeError:
            # TODO : find a way to check if the io_loop is running before trying to start it
            # this method to check if the loop is running is ugly
            pass