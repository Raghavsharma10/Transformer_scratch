def _serialize(self, convert_to_key_and_value, ignore_missing=False):
        """
        serialize model object to dictionary
        :param convert_to_key_and_value: function(field_name, value, property_detail) -> key, value
        :return:
        """

        serialized = {}
        properties = self._get_property_names(self)

        def get_property_detail(name):
            p = [p for p in self._property_details if p.name == name]
            return None if len(p) == 0 else p[0]

        for p in properties:
            pd = get_property_detail(p)
            value = self._property_to_field(p, pd)
            field_name = p if not pd else pd.to_field_name()

            if value is None or (ignore_missing and not value) or (pd and pd.unsent):
                continue
            else:
                key, value = convert_to_key_and_value(field_name, value, pd)
                if key:
                    serialized[key] = value

        return serialized