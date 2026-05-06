def load(self, host, exact_host_match=False):
        """ Load a config for a hostname or url.

        This method calls :func:`ftr_get_config` and :meth`append`
        internally. Refer to their docs for details on parameters.
        """

        # Can raise a SiteConfigNotFound, intentionally bubbled.
        config_string, host_string = ftr_get_config(host, exact_host_match)

        if config_string is None:
            LOGGER.error(u'Error while loading configuration.',
                         extra={'siteconfig': host_string})
            return

        self.append(ftr_string_to_instance(config_string))