def get_description(self):
        """Gets a description of this service implementation.

        return: (osid.locale.DisplayText) - a description
        compliance: mandatory - This method must be implemented.

        """
        return DisplayText({'text': profile.DESCRIPTION,
                            'languageTypeId': profile.LANGUAGETYPEID,
                            'scriptTypeId': profile.SCRIPTTYPEID,
                            'formatTypeId': profile.FORMATTYPEID})