def _set_supported_content_type(self, content_types_supported):
        """ Checks and sets the supported content types configuration value.
        """
        if not isinstance(content_types_supported, list):
            raise TypeError(("Settings 'READTIME_CONTENT_SUPPORT' must be"
                             "a list of content types."))

        self.content_type_supported = content_types_supported