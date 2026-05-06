def register_model(self, model, bundle):
        """
        Registers a bundle as the main bundle for a
        model. Used when we need to lookup urls by
        a model.
        """
        if model in self._model_registry:
            raise AlreadyRegistered('The model %s is already registered' \
                                     % model)

        if bundle.url_params:
            raise Exception("A primary model bundle cannot have dynamic \
                            url_parameters")

        self._model_registry[model] = bundle