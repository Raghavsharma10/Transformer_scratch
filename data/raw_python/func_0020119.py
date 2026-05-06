def update_or_create(self, model, **kwargs):
        '''Update or create a new instance of ``model``.

        This method can raise an exception if the ``kwargs`` dictionary
        contains field data that does not validate.

        :param model: a :class:`StdModel`
        :param kwargs: dictionary of parameters.
        :returns: A two elements tuple containing the instance and a boolean
            indicating if the instance was created or not.
        '''
        backend = self.model(model).backend
        return backend.execute(self._update_or_create(model, **kwargs))