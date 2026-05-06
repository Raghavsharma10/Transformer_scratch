def _set_values_on_model(self, model, values, fields=None):
        """
        Updates the values with the specified values.

        :param Model model: The sqlalchemy model instance
        :param dict values: The dictionary of attributes and
            the values to set.
        :param list fields: A list of strings indicating
            the valid fields. Defaults to self.fields.
        :return: The model with the updated
        :rtype: Model
        """
        fields = fields or self.fields
        for name, val in six.iteritems(values):
            if name not in fields:
                continue
            setattr(model, name, val)
        return model