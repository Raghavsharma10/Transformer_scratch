def register(self, ModelClass, form_field=None, widget=None, title=None, prefix=None, has_id_value=True):
        """
        Register a custom model with the ``AnyUrlField``.
        """
        if any(urltype.model == ModelClass for urltype in self._url_types):
            raise ValueError("Model is already registered: '{0}'".format(ModelClass))

        opts = ModelClass._meta
        opts = opts.concrete_model._meta

        if not prefix:
            # Store something descriptive, easier to lookup from raw database content.
            prefix = '{0}.{1}'.format(opts.app_label, opts.object_name.lower())
        if not title:
            title = ModelClass._meta.verbose_name

        if self.is_external_url_prefix(prefix):
            raise ValueError("Invalid prefix value: '{0}'.".format(prefix))
        if self[prefix] is not None:
            raise ValueError("Prefix is already registered: '{0}'".format(prefix))
        if form_field is not None and widget is not None:
            raise ValueError("Provide either a form_field or widget; use the widget parameter of the form field instead.")

        urltype = UrlType(ModelClass, form_field, widget, title, prefix, has_id_value)
        signals.post_save.connect(_on_model_save, sender=ModelClass)
        self._url_types.append(urltype)
        return urltype