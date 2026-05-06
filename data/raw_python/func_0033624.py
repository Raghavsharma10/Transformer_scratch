def handle_error(self, error, req, schema):
        """Handles errors during parsing. Aborts the current HTTP request and
        responds with a 422 error.
        """

        status_code = getattr(error, "status_code", self.DEFAULT_VALIDATION_STATUS)
        abort(status_code, exc=error, messages=error.messages, schema=schema)