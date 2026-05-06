def _parse_guild_info(self, info_container):
        """
        Parses the guild's general information and applies the found values.

        Parameters
        ----------
        info_container: :class:`bs4.Tag`
            The parsed content of the information container.
        """
        m = founded_regex.search(info_container.text)
        if m:
            description = m.group("desc").strip()
            self.description = description if description else None
            self.world = m.group("world")
            self.founded = parse_tibia_date(m.group("date").replace("\xa0", " "))
            self.active = "currently active" in m.group("status")