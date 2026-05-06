def _parse_api_value_list(self, values):
        """Return a list field compatible with the API."""
        try:
            return [v.to_api() for v in values]
        # Not models
        except AttributeError:
            return list(values)