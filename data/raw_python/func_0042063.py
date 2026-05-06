def _parse_guild_disband_info(self, info_container):
        """
        Parses the guild's disband info, if available.

        Parameters
        ----------
        info_container: :class:`bs4.Tag`
            The parsed content of the information container.
        """
        m = disband_regex.search(info_container.text)
        if m:
            self.disband_condition = m.group(2)
            self.disband_date = parse_tibia_date(m.group(1).replace("\xa0", " "))