def get_identifier_input(self, identifier_data):
        """Convert the various formats of input identifier_data into
        the proper json format expected by the ApiClient fetch method,
        which is a list of dicts."""

        identifier_input = []

        if isinstance(identifier_data, list) and len(identifier_data) > 0:
            # if list, convert each item in the list to json
            for address in identifier_data:
                identifier_input.append(self._convert_to_identifier_json(address))
        else:
            identifier_input.append(self._convert_to_identifier_json(identifier_data))

        return identifier_input