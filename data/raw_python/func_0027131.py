def handle_response_for_connection(self, should_post=False):
        """ Check if the response succeed or not.

            In case of error, this method also print messages and set
            an array of errors in the response object.

            Returns:
                Returns True if the response has succeed, False otherwise
        """

        status_code = self._response.status_code
        data = self._response.data

        # TODO : Get errors in response data after bug fix : http://mvjira.mv.usa.alcatel.com/browse/VSD-2735
        if data and 'errors' in data:
            self._response.errors = data['errors']

        if status_code in [HTTP_CODE_SUCCESS, HTTP_CODE_CREATED, HTTP_CODE_EMPTY]:
            return True

        if status_code == HTTP_CODE_MULTIPLE_CHOICES:
            return False

        if status_code in [HTTP_CODE_PERMISSION_DENIED, HTTP_CODE_UNAUTHORIZED]:

            if not should_post:
                return True

            return False

        if status_code in [HTTP_CODE_CONFLICT, HTTP_CODE_NOT_FOUND, HTTP_CODE_BAD_REQUEST, HTTP_CODE_METHOD_NOT_ALLOWED, HTTP_CODE_PRECONDITION_FAILED, HTTP_CODE_SERVICE_UNAVAILABLE]:
            if not should_post:
                return True

            return False

        if status_code == HTTP_CODE_INTERNAL_SERVER_ERROR:

            return False

        if status_code == HTTP_CODE_ZERO:
            bambou_logger.error("NURESTConnection: Connection error with code 0. Sending NUNURESTConnectionFailureNotification notification and exiting.")
            return False

        bambou_logger.error("NURESTConnection: Report this error, because this should not happen: %s" % self._response)
        return False