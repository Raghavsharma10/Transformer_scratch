def applicable_models(self):
        """
        Returns a list of model classes that subclass Page
        and include a "tags" field.

        :rtype: list.
        """
        Page        = apps.get_model('wagtailcore', 'Page')
        applicable  = []

        for model in apps.get_models():
            meta    = getattr(model, '_meta')
            fields  = meta.get_all_field_names()

            if issubclass(model, Page) and 'tags' in fields:
                applicable.append(model)

        return applicable