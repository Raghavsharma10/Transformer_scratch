def get_display_name(self):
        """Gets a display name for this service implementation.

        return: (osid.locale.DisplayText) - a display name
        *compliance: mandatory -- This method must be implemented.*

        """
        return DisplayText(
            text=profile.DISPLAYNAME,
            language_type=Type(**profile.LANGUAGETYPE),
            script_type=Type(**profile.SCRIPTTYPE),
            format_type=Type(**profile.FORMATTYPE))