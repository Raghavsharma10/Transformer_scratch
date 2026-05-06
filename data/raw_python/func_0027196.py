def register_model(cls, model):
        """
            Register a model class according to its remote name

            Args:
                model: the model to register
        """

        rest_name = model.rest_name
        resource_name = model.resource_name

        if rest_name not in cls._model_rest_name_registry:
            cls._model_rest_name_registry[rest_name] = [model]
            cls._model_resource_name_registry[resource_name] = [model]

        elif model not in cls._model_rest_name_registry[rest_name]:
            cls._model_rest_name_registry[rest_name].append(model)
            cls._model_resource_name_registry[resource_name].append(model)