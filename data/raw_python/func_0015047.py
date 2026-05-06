def serialize_model(self, model, field_dict=None):
        """
        Takes a model and serializes the fields provided into
        a dictionary.

        :param Model model: The Sqlalchemy model instance to serialize
        :param dict field_dict: The dictionary of fields to return.
        :return: The serialized model.
        :rtype: dict
        """
        response = self._serialize_model_helper(model, field_dict=field_dict)
        return make_json_safe(response)