def _list_response(self, response):
        """
        This method check if the response is a dict and wrap it into a list.
        If the response is already a list, it returns the response directly.
        This workaround is necessary because the API doesn't return a list
        if only one item is found.
        """
        if type(response) is list:
            return response
        if type(response) is dict:
            return [response]