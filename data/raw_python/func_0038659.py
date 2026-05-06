def setup_managers(self):
        """
        Allows to access manager by model name - it is convenient, because HasOffers returns model names in responses.
        """
        self._managers = {}
        for manager_class in MODEL_MANAGERS:
            instance = manager_class(self)
            if not instance.forbid_registration \
                    and not isinstance(instance, ApplicationManager) or instance.__class__ is ApplicationManager:
                # Descendants of ``ApplicationManager`` shouldn't be present in API instance.  They are controlled by
                # Application controller. The manager itself, on the other hand, should.
                setattr(self, instance.name, instance)
            if instance.model:
                self._managers[instance.model.__name__] = instance
            if instance.model_aliases:
                for alias in instance.model_aliases:
                    self._managers[alias] = instance