def _request(self, method, url, query_or_data=None, **kwargs):
        """
        Wrapper for the HTTP requests,
        rate limit backoff is handled here,
        responses are processed with ResourceBuilder.
        """

        if query_or_data is None:
            query_or_data = {}

        request_method = getattr(self, '_http_{0}'.format(method))
        response = retry_request(self)(request_method)(url, query_or_data, **kwargs)

        if self.raw_mode:
            return response

        if response.status_code >= 300:
            error = get_error(response)
            if self.raise_errors:
                raise error
            return error

        # Return response object on NoContent
        if response.status_code == 204 or not response.text:
            return response

        return ResourceBuilder(
            self,
            self.default_locale,
            response.json()
        ).build()