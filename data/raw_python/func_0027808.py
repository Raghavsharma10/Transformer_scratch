def setup_logging(self, defaults=None):
        """
        Set up logging via :func:`logging.config.fileConfig`.

        Defaults are specified for the special ``__file__`` and ``here``
        variables, similar to PasteDeploy config loading. Extra defaults can
        optionally be specified as a dict in ``defaults``.

        :param defaults: The defaults that will be used when passed to
            :func:`logging.config.fileConfig`.
        :return: ``None``.

        """
        if "loggers" in self.get_sections():
            defaults = self._get_defaults(defaults)
            fileConfig(self.uri.path, defaults, disable_existing_loggers=False)

        else:
            logging.basicConfig()