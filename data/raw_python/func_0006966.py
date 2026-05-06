def validate_request_success(
        response_text, request_url, status_code, expected_status_code
    ):
        """Validates that a request was successful.

        Args:
            response_text (str): The response body of the request.
            request_url (str): The URL the request was made at.
            status_code (int): The status code of the response.
            expected_status_code (int): The expected status code of the
                response.

        Raises:
            :class:`saltant.exceptions.BadHttpRequestError`: The HTTP
                request failed.
        """
        try:
            assert status_code == expected_status_code
        except AssertionError:
            msg = (
                "Request to {url} failed with status {status_code}:\n"
                "The reponse from the request was as follows:\n\n"
                "{content}"
            ).format(
                url=request_url, status_code=status_code, content=response_text
            )
            raise BadHttpRequestError(msg)