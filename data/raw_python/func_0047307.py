def get_description(self):
        """Creates a description"""
        return DisplayText(text='Agent representing ' + str(self.id_),
                           language_type=DEFAULT_LANGUAGE_TYPE,
                           script_type=DEFAULT_SCRIPT_TYPE,
                           format_type=DEFAULT_FORMAT_TYPE,)