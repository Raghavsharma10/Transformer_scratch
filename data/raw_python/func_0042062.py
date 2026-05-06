def _parse_application_info(self, info_container):
        """
        Parses the guild's application info.

        Parameters
        ----------
        info_container: :class:`bs4.Tag`
            The parsed content of the information container.
        """
        m = applications_regex.search(info_container.text)
        if m:
            self.open_applications = m.group(1) == "opened"