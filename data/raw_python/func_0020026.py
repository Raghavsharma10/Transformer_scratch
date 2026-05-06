def _deserialize(cls, json_body, get_value_and_type):
        """
        deserialize json to model
        :param json_body: json data
        :param get_value_and_type: function(f: json_field) -> value, field_type_string(see FieldType)
        :return:
        """

        instance = cls()
        is_set = False
        properties = cls._get_property_names(instance)

        def get_property_detail(name):
            p = [p for p in instance._property_details if p.name == name or p.field_name == name]
            return None if len(p) == 0 else p[0]

        for k in json_body:
            field = json_body[k]
            pd = get_property_detail(k)
            pn = k if not pd else pd.to_property_name(k)
            if pn in properties:
                v, t = get_value_and_type(field)
                initial_value = getattr(instance, pn)
                value = instance._field_to_property(v, t, pd, initial_value)
                setattr(instance, pn, value)
                is_set = True

        return instance if is_set else None