def set_primary_parameters(self, **kwargs):
        """Set all primary parameters at once."""
        given = sorted(kwargs.keys())
        required = sorted(self._PRIMARY_PARAMETERS)
        if given == required:
            for (key, value) in kwargs.items():
                setattr(self, key, value)
        else:
            raise ValueError(
                'When passing primary parameter values as initialization '
                'arguments of the instantaneous unit hydrograph class `%s`, '
                'or when using method `set_primary_parameters, one has to '
                'to define all values at once via keyword arguments.  '
                'But instead of the primary parameter names `%s` the '
                'following keywords were given: %s.'
                % (objecttools.classname(self),
                   ', '.join(required), ', '.join(given)))