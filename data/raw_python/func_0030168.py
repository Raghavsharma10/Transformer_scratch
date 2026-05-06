def register_model(cls, ModelClass, form_field=None, widget=None, title=None, prefix=None):
        """
        Register a model to use in the URL field.

        This function needs to be called once for every model
        that should be selectable in the URL field.

        :param ModelClass: The model to register.
        :param form_field: The form field class used to render the field. This can be a lambda for lazy evaluation.
        :param widget: The widget class, can be used instead of the form field.
        :param title: The title of the model, by default it uses the models ``verbose_name``.
        :param prefix: A custom prefix for the model in the serialized database format. By default it uses "appname.modelname".
        """
        cls._static_registry.register(ModelClass, form_field, widget, title, prefix)