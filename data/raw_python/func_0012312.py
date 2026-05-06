def run(self, host="localhost", port=8000, shutdown_timeout=60.0, **kwargs):
        """
            This function starts the service's network intefaces.

            Args:
                port (int): The port for the http server.
        """
        print("Running service on http://localhost:%i. " % port + \
                                            "Press Ctrl+C to terminate.")

        # apply the configuration to the service config
        self.config.port = port
        self.config.host = host

        # start the loop
        try:
            # if an event broker has been created for this service
            if self.event_broker:
                # start the broker
                self.event_broker.start()
                # announce the service
                self.loop.run_until_complete(self.announce())

            # the handler for the http server
            http_handler = self.app.make_handler()
            # create an asyncio server
            self._http_server = self.loop.create_server(http_handler, host, port)

            # grab the handler for the server callback
            self._server_handler = self.loop.run_until_complete(self._http_server)
            # start the event loop
            self.loop.run_forever()

        # if the user interrupted the server
        except KeyboardInterrupt:
            # keep going
            pass

        # when we're done
        finally:
            try:
                # clean up the service
                self.cleanup()
            # if we end up closing before any variables get assigned
            except UnboundLocalError:
                # just ignore it (there was nothing to close)
                pass

            # close the event loop
            self.loop.close()