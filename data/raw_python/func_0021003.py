def search(self, **kwargs):
        """Searches for files/folders

        Args:
            \*\*kwargs (dict): A dictionary containing necessary parameters
                            (check https://developers.box.com/docs/#search for
                            list of parameters)

        Returns:
            dict. Response from Box.

        Raises:
            BoxError: An error response is returned from Box (status_code >= 400).

            BoxHttpResponseError: Response from Box is malformed.

            requests.exceptions.*: Any connection related problem.
        """
        query_string = {}
        for key, value in kwargs.iteritems():
            query_string[key] = value
        return self.__request("GET","search",querystring=query_string)