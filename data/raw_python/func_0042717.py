def register_models(self, *models, **kwargs):
        """
        Register multiple models with the same
        arguments.

        Calls register for each argument passed along with
        all keyword arguments.
        """

        for model in models:
            self.register(model, **kwargs)