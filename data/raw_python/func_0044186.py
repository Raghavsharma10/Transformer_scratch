def get_backend_engine(self, name, **kwargs):
        """
        Get backend engine from given name.

        Args:
            (string): Path to validate.

        Raises:
            boussole.exceptions.SettingsBackendError: If given backend name
                does not match any available engine.

        Returns:
            object: Instance of selected backend engine.
        """
        if name not in self._engines:
            msg = "Given settings backend is unknowed: {}"
            raise SettingsBackendError(msg.format(name))

        return self._engines[name](**kwargs)