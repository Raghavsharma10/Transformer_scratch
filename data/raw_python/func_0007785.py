def is_acceptable(self, response, request_params):
        """
        Override this method to create a different definition of
        what kind of response is acceptable.
        If `bool(the_return_value) is False` then an `HTTPServiceError`
        will be raised.

        For example, you might want to assert that the body must be empty,
        so you could return `len(response.content) == 0`.

        In the default implementation, a response is acceptable
        if and only if the response code is either
        less than 300 (typically 200, i.e. OK) or if it is in the
        `expected_response_codes` parameter in the constructor.
        """
        expected_codes = request_params.get('expected_response_codes', [])
        return response.is_ok or response.status_code in expected_codes