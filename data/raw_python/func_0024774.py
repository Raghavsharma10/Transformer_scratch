def evaluate_response(json_response):
        """Evaluate rest response."""
        if 'errors' in json_response and json_response['errors']:
            Interface.evaluate_errors(json_response)
        elif 'result' not in json_response:
            raise PyVLXException('no element result  found in response: {0}'.format(json.dumps(json_response)))
        elif not json_response['result']:
            raise PyVLXException('Request failed {0}'.format(json.dumps(json_response)))