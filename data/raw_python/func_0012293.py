def _from_type(self, config):
        """
            This method converts a type into a dict.
        """
        def is_user_attribute(attr):
            return (
                not attr.startswith('__') and
                not isinstance(getattr(config, attr), collections.abc.Callable)
            )

        return {attr: getattr(config, attr) for attr in dir(config) \
                                                    if is_user_attribute(attr)}