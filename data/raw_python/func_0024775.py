def evaluate_errors(json_response):
        """Evaluate rest errors."""
        if 'errors' not in json_response or \
           not isinstance(json_response['errors'], list) or \
           not json_response['errors'] or \
           not isinstance(json_response['errors'][0], int):
            raise PyVLXException('Could not evaluate errors {0}'.format(json.dumps(json_response)))

        # unclear if response may contain more errors than one. Taking the first.
        first_error = json_response['errors'][0]

        if first_error in [402, 403, 405, 406]:
            raise InvalidToken(first_error)

        raise PyVLXException('Unknown error code {0}'.format(first_error))