def execute_request(self, url, http_method, query_params, post_data):
        """Makes a request to the specified url endpoint with the
        specified http method, params and post data.

        Args:
            url (string): The url to the API without query params.
                          Example: "https://api.housecanary.com/v2/property/value"
            http_method (string): The http method to use for the request.
            query_params (dict): Dictionary of query params to add to the request.
            post_data: Json post data to send in the body of the request.

        Returns:
            The result of calling this instance's OutputGenerator process_response method
            on the requests.Response object.
            If no OutputGenerator is specified for this instance, returns the requests.Response.
        """

        response = requests.request(http_method, url, params=query_params,
                                    auth=self._auth, json=post_data,
                                    headers={'User-Agent': USER_AGENT})

        if isinstance(self._output_generator, str) and self._output_generator.lower() == "json":
            # shortcut for just getting json back
            return response.json()
        elif self._output_generator is not None:
            return self._output_generator.process_response(response)
        else:
            return response