def unregister_model(self, model):
        """
        Unregisters the given model.
        """
        if model not in self._model_registry:
            raise NotRegistered('The model %s is not registered' % model)

        del self._model_registry[model]