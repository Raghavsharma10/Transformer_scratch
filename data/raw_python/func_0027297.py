def add_function(self, function):
        """
        Registers the function to the server's default fixed function manager.
        """
        #noinspection PyTypeChecker
        if not len(self.settings.FUNCTION_MANAGERS):
            raise ConfigurationError(
                'Where have the default function manager gone?!')
        self.settings.FUNCTION_MANAGERS[0].add_function(function)