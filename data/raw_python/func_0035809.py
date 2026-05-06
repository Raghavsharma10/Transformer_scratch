def _to_api_value(self, attribute_type, value):
        """Return a parsed value for the API."""

        if not value:
            return None

        if isinstance(attribute_type, properties.Instance):
            return value.to_api()

        if isinstance(attribute_type, properties.List):
            return self._parse_api_value_list(value)

        return attribute_type.serialize(value)