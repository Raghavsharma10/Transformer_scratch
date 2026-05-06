def model_function(self, model_name, version, func_name):
        """Return the model-specific caching function."""
        assert func_name in ('serializer', 'loader', 'invalidator')
        name = "%s_%s_%s" % (model_name.lower(), version, func_name)
        return getattr(self, name)