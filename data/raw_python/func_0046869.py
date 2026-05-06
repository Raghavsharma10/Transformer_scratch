def get_display_name(self):
        """Gets a display name for this service implementation.

        return: (osid.locale.DisplayText) - a display name
        compliance: mandatory - This method must be implemented.

        """
        return DisplayText({'text': profile.DISPLAYNAME,
                            'languageTypeId': profile.LANGUAGETYPEID,
                            'scriptTypeId': profile.SCRIPTTYPEID,
                            'formatTypeId': profile.FORMATTYPEID})