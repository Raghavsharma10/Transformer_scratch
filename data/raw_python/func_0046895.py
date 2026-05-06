def _request(self, method, *args, **kwargs):
        """Make a request with automatic pagination handling

        Args:
            method (str): A dot delimited string indicating the method to call.  Example: 'Machines.List'
            *args: Passed directly to the method being called.
            **kwargs: Passed directly to the method being called.
                        Note: This method will inject the 'nextPageToken' key into `**kwargs` as needed to handle
                        pagination overwriting any value specified by the caller.  If you wish to handle pagination
                        manually use the `_single_request` method


        Yields:
            dict: The next page of responses from the method called.


        Raises:
            fleet.v1.errors.APIError: Fleet returned a response code >= 400

        """

        # This is set to False and not None so that the while loop below will execute at least once
        next_page_token = False

        while next_page_token is not None:
            # If bool(next_page_token), then include it in the request
            # We do this so we don't pass it in the initial request as we set it to False above
            if next_page_token:
                kwargs['nextPageToken'] = next_page_token

            # Make the request
            response = self._single_request(method, *args, **kwargs)

            # If there is a token for another page in the response, capture it for the next loop iteration
            # If not, we set it to None so that the loop will terminate
            next_page_token = response.get('nextPageToken', None)

            # Return the current response
            yield response