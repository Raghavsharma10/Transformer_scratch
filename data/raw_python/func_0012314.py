def add_http_endpoint(self, url, request_handler):
        """
            This method provides a programatic way of added invidual routes
            to the http server.

            Args:
                url (str): the url to be handled by the request_handler
                request_handler (nautilus.network.RequestHandler): The request handler
        """
        self.app.router.add_route('*', url, request_handler)