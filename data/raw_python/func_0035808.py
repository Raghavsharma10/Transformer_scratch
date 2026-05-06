def to_api(self):
        """Return a dictionary to send to the API.

        Returns:
            dict: Mapping representing this object that can be sent to the
             API.
        """
        vals = {}
        for attribute, attribute_type in self._props.items():
            prop = getattr(self, attribute)
            vals[self._to_camel_case(attribute)] = self._to_api_value(
                attribute_type, prop,
            )
        return vals