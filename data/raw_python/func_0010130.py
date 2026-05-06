def load_raw(cls, model_fn, schema, *args, **kwargs):
        """
        Loads a trained classifier from the raw Weka model format.
        Must specify the model schema and classifier name, since
        these aren't currently deduced from the model format.
        """
        c = cls(*args, **kwargs)
        c.schema = schema.copy(schema_only=True)
        c._model_data = open(model_fn, 'rb').read()
        return c