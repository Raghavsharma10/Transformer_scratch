def applicable_models(self):
        """
        Returns a list of model classes that subclass Page.

        :rtype: list.
        """
        Page        = apps.get_model('wagtailcore', 'Page')
        applicable  = []

        for model in apps.get_models():
            if issubclass(model, Page):
                applicable.append(model)

        return applicable