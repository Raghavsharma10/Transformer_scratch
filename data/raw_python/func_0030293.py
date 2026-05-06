def coerce(cls, key, value):
        """Convert plain list to MutationList."""

        if isinstance(value, string_types):
            value = value.strip()
            if value[0] == '[':  # It's json encoded, probably
                try:
                    value = json.loads(value)
                except ValueError:
                    raise ValueError("Failed to parse JSON: '{}' ".format(value))
            else:
                value = value.split(',')

        if not value:
            value = []

        self = MutationList((MutationObj.coerce(key, v) for v in value))
        self._key = key
        return self