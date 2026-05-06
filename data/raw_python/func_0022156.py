def _catch_nonjson_streamresponse(rawresponse):
        """
        Validate a streamed response is JSON. Return a Python dictionary either way.


        **Parameters:**

          - **rawresponse:** Streamed Response from Requests.

        **Returns:** Dictionary
        """
        # attempt to load response for return.
        try:
            response = json.loads(rawresponse)
        except (ValueError, TypeError):
            if rawresponse:
                response = {
                    '_error': [
                        {
                            'message': 'Response not in JSON format.',
                            'data': rawresponse,
                        }
                    ]
                }
            else:
                # in case of null response, return empty dict.
                response = {}

        return response