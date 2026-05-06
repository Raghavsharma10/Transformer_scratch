def add_operation(self, operation_type, operation, mode=None):
        """Add an operation to the version

        :param mode: Name of the mode in which the operation is executed
        :type mode: str
        :param operation_type: one of 'pre', 'post'
        :type operation_type: str
        :param operation: the operation to add
        :type operation: :class:`marabunta.model.Operation`
        """
        version_mode = self._get_version_mode(mode=mode)
        if operation_type == 'pre':
            version_mode.add_pre(operation)
        elif operation_type == 'post':
            version_mode.add_post(operation)
        else:
            raise ConfigurationError(
                u"Type of operation must be 'pre' or 'post', got %s" %
                (operation_type,)
            )