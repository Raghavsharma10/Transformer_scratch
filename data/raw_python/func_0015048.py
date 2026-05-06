def _serialize_model_helper(self, model, field_dict=None):
        """
        A recursive function for serializing a model
        into a json ready format.
        """
        field_dict = field_dict or self.dot_field_list_to_dict()
        if model is None:
            return None

        if isinstance(model, Query):
            model = model.all()

        if isinstance(model, (list, set)):
            return [self.serialize_model(m, field_dict=field_dict) for m in model]

        model_dict = {}
        for name, sub in six.iteritems(field_dict):
            value = getattr(model, name)
            if sub:
                value = self.serialize_model(value, field_dict=sub)
            model_dict[name] = value
        return model_dict