def _handle_response(response, **kwargs) -> XMLResponse:
        """Requests HTTP Response handler. Attaches .html property to
        class:`requests.Response <requests.Response>` objects.
        """
        if not response.encoding:
            response.encoding = DEFAULT_ENCODING

        return response