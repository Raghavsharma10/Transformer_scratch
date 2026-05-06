def post(self, url, post_data, query_params=None):
        """Makes a POST request to the specified url endpoint.

        Args:
            url (string): The url to the API without query params.
                          Example: "https://api.housecanary.com/v2/property/value"
            post_data: Json post data to send in the body of the request.
            query_params (dict): Optional. Dictionary of query params to add to the request.

        Returns:
            The result of calling this instance's OutputGenerator process_response method
            on the requests.Response object.
            If no OutputGenerator is specified for this instance, returns the requests.Response.
        """
        if query_params is None:
            query_params = {}

        return self.execute_request(url, "POST", query_params, post_data)