def extract_data_from_response(self, response, data_key=None):
        """Given a response and an optional data_key should return a dictionary of data returned as part of the response."""
        response_json_data = response.json()
        # Seems to be two types of response, a dict with keys and then lists of data or a flat list data with no key.
        if type(response_json_data) == list:
            # Return the data
            return response_json_data
        elif type(response_json_data) == dict:
            if data_key is None:
                return response_json_data
            else:
                return response_json_data[data_key]
        else:
            raise CanvasAPIError(response)