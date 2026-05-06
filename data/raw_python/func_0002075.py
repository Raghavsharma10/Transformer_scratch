def from_requests_error(cls, err):
        """
        Raises a subclass of ServerError based on the HTTP response code.
        """
        import requests
        if isinstance(err, requests.HTTPError):
            status_code = err.response.status_code
            return HTTP_ERRORS.get(status_code, cls)(error=err, response=err.response)
        else:
            return cls(error=err)