def _adapt_response(self, response):
        """Convert error responses to standardized ErrorDetails."""
        if 'application/json' in response.headers['content-type']:
            body = response.json()
            status = response.status_code

            if body.get('error'):
                return self._simple_response_to_error_adapter(status, body)

        raise UnknownHttpError(response)