def serialize_model(self, value):
        """
        Serializes a model and all of its prefetched foreign keys
        :param value:
        :return:
        """

        # Check if the context value is a model
        if not isinstance(value, models.Model):
            return value

        # Serialize the model
        serialized_model = model_to_dict(value)

        # Check the model for cached foreign keys
        for model_field, model_value in serialized_model.items():
            model_state = value._state

            # Django >= 2
            if hasattr(model_state, 'fields_cache'):  # pragma: no cover
                if model_state.fields_cache.get(model_field):
                    serialized_model[model_field] = model_state.fields_cache.get(model_field)
            else:  # pragma: no cover
                # Django < 2
                cache_field = '_{0}_cache'.format(model_field)
                if hasattr(value, cache_field):
                    serialized_model[model_field] = getattr(value, cache_field)

        # Return the serialized model
        return self.serialize_value(serialized_model)