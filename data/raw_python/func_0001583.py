def _register(self, form_class, check_middleware=True):
        """
        Register a config form into the registry

        :param object form_class: The form class to register.\
        Must be an instance of :py:class:`djconfig.forms.ConfigForm`
        :param bool check_middleware: Check\
        :py:class:`djconfig.middleware.DjConfigMiddleware`\
        is registered into ``settings.MIDDLEWARE_CLASSES``. Default True
        """
        if not issubclass(form_class, _ConfigFormBase):
            raise ValueError(
                "The form does not inherit from `forms.ConfigForm`")

        self._registry.add(form_class)

        if check_middleware:
            _check_backend()