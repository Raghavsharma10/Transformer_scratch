def _parse_guild_guildhall(self, info_container):
        """
        Parses the guild's guildhall info.

        Parameters
        ----------
        info_container: :class:`bs4.Tag`
            The parsed content of the information container.
        """
        m = guildhall_regex.search(info_container.text)
        if m:
            paid_until = parse_tibia_date(m.group("date").replace("\xa0", " "))
            self.guildhall = GuildHouse(m.group("name"), self.world, paid_until_date=paid_until)