def register(self, *model_list, **options):
        """
        Registers the given model(s) with the given databrowse site.

        The model(s) should be Model classes, not instances.

        If a databrowse class isn't given, it will use DefaultModelDatabrowse
        (the default databrowse options).

        If a model is already registered, this will raise AlreadyRegistered.
        """
        databrowse_class = options.pop('databrowse_class',
                                       DefaultModelDatabrowse)
        for model in model_list:
            if model in self.registry:
                raise AlreadyRegistered('The model %s is already registered' %
                                        model.__name__)
            self.registry[model] = databrowse_class