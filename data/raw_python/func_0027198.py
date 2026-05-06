def get_first_model_with_resource_name(cls, resource_name):
        """ Get the first model corresponding to a resource_name

            Args:
                resource_name: the resource name
        """

        models = cls.get_models_with_resource_name(resource_name)

        if len(models) > 0:
            return models[0]

        return None