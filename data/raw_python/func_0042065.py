def _parse_guild_homepage(self, info_container):
        """
        Parses the guild's homepage info.

        Parameters
        ----------
        info_container: :class:`bs4.Tag`
            The parsed content of the information container.
        """
        m = homepage_regex.search(info_container.text)
        if m:
            self.homepage = m.group(1)