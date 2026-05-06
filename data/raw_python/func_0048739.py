def get_model(self):
        """
        Return the class Model used by this Agnocomplete
        """
        if hasattr(self, 'model') and self.model:
            return self.model
        # Give me a "none" queryset
        try:
            none = self.get_queryset().none()
            return none.model
        except Exception:
            raise ImproperlyConfigured(
                "Integrator: Unable to determine the model with this queryset."
                " Please add a `model` property")