def from_dict(cls, data, model):
        """ Generate map of `fieldName: clsInstance` from dict.

        :param data: Dict where keys are field names and values are
            new values of field.
        :param model: Model class to which fields from :data: belong.
        """
        model_provided = model is not None
        result = {}
        for name, new_value in data.items():
            kwargs = {
                'name': name,
                'new_value': new_value,
            }
            if model_provided:
                kwargs['params'] = model.get_field_params(name)
            result[name] = cls(**kwargs)
        return result