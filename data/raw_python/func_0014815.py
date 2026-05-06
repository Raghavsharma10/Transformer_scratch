def to_internal_value(self, data):
        """Because we allow template ID string values, where
        serializers normally expect a dict
        """
        converted_data = _convert_template_id_to_dict(data)
        return super(TemplateSerializer, self)\
            .to_internal_value(converted_data)