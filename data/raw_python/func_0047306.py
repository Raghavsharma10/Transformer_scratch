def get_display_name(self):
        """Creates a display name"""
        return DisplayText(text=self.id_.get_identifier(),
                           language_type=DEFAULT_LANGUAGE_TYPE,
                           script_type=DEFAULT_SCRIPT_TYPE,
                           format_type=DEFAULT_FORMAT_TYPE,)