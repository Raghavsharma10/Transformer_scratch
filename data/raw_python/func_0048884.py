def get_description(self):
        """Gets a description of this service implementation.

        return: (osid.locale.DisplayText) - a description
        *compliance: mandatory -- This method must be implemented.*

        """
        return DisplayText(
            text=profile.DESCRIPTION,
            language_type=Type(**profile.LANGUAGETYPE),
            script_type=Type(**profile.SCRIPTTYPE),
            format_type=Type(**profile.FORMATTYPE))