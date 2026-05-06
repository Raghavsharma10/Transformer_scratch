def _simple_response_to_error_adapter(self, status, original_body):
        """Convert a single error response."""
        meta = original_body.get('error')
        e = []

        if 'error_detail' in original_body:
            errors = original_body.get('error_detail')

            for error in errors:
                if type(error) == dict:
                    for parameter, title in error.iteritems():
                        e.append(ErrorDetails(parameter, title))
        elif 'error_description' in original_body:
            e.append(original_body.get('error_description'))

        return e, meta