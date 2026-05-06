def initialize_settings(self, sender):
        """ Initializes ReadTimeParser with configuration values set by the
        site author.
        """
        try:
            self.initialized = True

            settings_content_types = sender.settings.get(
                'READTIME_CONTENT_SUPPORT', self.content_type_supported)
            self._set_supported_content_type(settings_content_types)

            lang_settings = sender.settings.get(
                'READTIME_WPM', self.lang_settings)
            self._set_lang_settings(lang_settings)
        except Exception as e:
            raise Exception("ReadTime Plugin: %s" % str(e))