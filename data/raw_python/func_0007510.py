def handle_error(self, error, response):
        """
        Redirects the client in case an error in the auth process occurred.
        """
        query_params = {"error": error.error}

        query = urlencode(query_params)

        location = "%s?%s" % (self.client.redirect_uri, query)

        response.status_code = 302
        response.body = ""
        response.add_header("Location", location)

        return response