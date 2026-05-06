def cleanup(self):
        """
            This function is called when the service has finished running
            regardless of intentionally or not.
        """

        # if an event broker has been created for this service
        if self.event_broker:
            # stop the event broker
            self.event_broker.stop()
        # attempt
        try:
            # close the http server
            self._server_handler.close()
            self.loop.run_until_complete(self._server_handler.wait_closed())
            self.loop.run_until_complete(self._http_handler.finish_connections(shutdown_timeout))

        # if there was no handler
        except AttributeError:
            # keep going
            pass

        # more cleanup
        self.loop.run_until_complete(self.app.shutdown())
        self.loop.run_until_complete(self.app.cleanup())