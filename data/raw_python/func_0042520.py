def update_rel_to(self, klass):
        """
        If we have a string for a model, see if we know about it yet,
        if so use it directly otherwise take the lazy approach.
        This check is needed because this is called before
        the main M2M field contribute to class is called.
        """

        if isinstance(self.remote_field.to, basestring):
            relation = self.remote_field.to
            try:
                app_label, model_name = relation.split(".")
            except ValueError:
                # If we can't split, assume a model in current app
                app_label = klass._meta.app_label
                model_name = relation

            model = None
            try:
                model = klass._meta.apps.get_registered_model(app_label, model_name)
            # For django < 1.6
            except AttributeError:
                model = models.get_model(
                    app_label, model_name,
                    seed_cache=False, only_installed=False)
            except LookupError:
                pass

            if model:
                self.remote_field.model = model