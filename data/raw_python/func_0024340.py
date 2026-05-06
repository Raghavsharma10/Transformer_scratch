def run(self, host, port, debug=True, validate_requests=True):
        """Utility method to quickly get a server up and running.

        :param debug: turns on Werkzeug debugger, code reloading, and full
            logging.
        :param validate_requests: whether or not to ensure that requests are
            sent by Amazon. This can be usefulfor manually testing the server.
        """

        if debug:
            # Turn on all alexandra log output
            logging.basicConfig(level=logging.DEBUG)

        app = self.create_wsgi_app(validate_requests)
        run_simple(host, port, app, use_reloader=debug, use_debugger=debug)