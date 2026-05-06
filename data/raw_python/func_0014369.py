def _delete(self, url, **kwargs):
        """
        Wrapper for the HTTP DELETE request.
        """

        response = retry_request(self)(self._http_delete)(url, **kwargs)

        if self.raw_mode:
            return response

        if response.status_code >= 300:
            error = get_error(response)
            if self.raise_errors:
                raise error
            return error

        return response