def has_succeed(self):
        """ Check if the connection has succeed

            Returns:
                Returns True if connection has succeed.
                False otherwise.

        """
        status_code = self._response.status_code

        if status_code in [HTTP_CODE_ZERO, HTTP_CODE_SUCCESS, HTTP_CODE_CREATED, HTTP_CODE_EMPTY, HTTP_CODE_MULTIPLE_CHOICES]:
            return True

        if status_code in [HTTP_CODE_BAD_REQUEST, HTTP_CODE_UNAUTHORIZED, HTTP_CODE_PERMISSION_DENIED, HTTP_CODE_NOT_FOUND, HTTP_CODE_METHOD_NOT_ALLOWED, HTTP_CODE_CONNECTION_TIMEOUT, HTTP_CODE_CONFLICT, HTTP_CODE_PRECONDITION_FAILED, HTTP_CODE_INTERNAL_SERVER_ERROR, HTTP_CODE_SERVICE_UNAVAILABLE]:
            return False

        raise Exception('Unknown status code %s.', status_code)