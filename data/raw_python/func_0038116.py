def from_model(cls, model_instance, default_value=False, **kwargs):
        """
        wrapper for Model's get_attribute_filter
        """

        if not isinstance(model_instance, DataCollection):
            raise TypeError("model_instance must be a subclass of \
                prestans.types.DataCollection, %s given" % (model_instance.__class__.__name__))
        elif isinstance(model_instance, Array) and model_instance.is_scalar:
            return AttributeFilter(is_array_scalar=True)
        attribute_filter_instance = model_instance.get_attribute_filter(default_value)

        # kwargs support
        for name, value in iter(kwargs.items()):
            if name in attribute_filter_instance:
                setattr(attribute_filter_instance, name, value)
            else:
                raise KeyError(name)

        return attribute_filter_instance