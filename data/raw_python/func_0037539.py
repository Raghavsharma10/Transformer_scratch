def start(self):
        """ Start the MQTT client. """
        self.thread_handler.run(target=self.start_blocking)
        self.thread_handler.start_run_loop()