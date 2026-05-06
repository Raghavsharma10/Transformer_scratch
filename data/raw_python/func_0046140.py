def get_license(self):
        """Gets the terms of usage.

        An empty license means the terms are unknown.

        return: (osid.locale.DisplayText) - the license
        *compliance: mandatory -- This method must be implemented.*

        """
        if 'license' in self.my_osid_object._my_map:
            license_text = self.my_osid_object._my_map['license']
            return DisplayText(display_text_map=license_text)
        return DisplayText(text='',
                           language_type=DEFAULT_LANGUAGE_TYPE,
                           format_type=DEFAULT_FORMAT_TYPE,
                           script_type=DEFAULT_SCRIPT_TYPE)