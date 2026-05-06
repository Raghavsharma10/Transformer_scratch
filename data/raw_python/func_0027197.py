def get_first_model_with_rest_name(cls, rest_name):
        """ Get the first model corresponding to a rest_name

            Args:
                rest_name: the rest name
        """

        models = cls.get_models_with_rest_name(rest_name)

        if len(models) > 0:
            return models[0]

        return None