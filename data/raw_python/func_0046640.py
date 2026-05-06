def _caching_enabled(self):
        """Returns True if caching is enabled per configuration, false otherwise."""
        try:
            config = self._runtime.get_configuration()
            parameter_id = Id('parameter:useCachingForQualifierIds@json')
            if config.get_value_by_parameter(parameter_id).get_boolean_value():
                return True
            else:
                return False
        except (AttributeError, KeyError, errors.NotFound):
            return False