def _parse_killer(cls, killer):
        """Parses a killer into a dictionary.

        Parameters
        ----------
        killer: :class:`str`
            The killer's raw HTML string.

        Returns
        -------
        :class:`dict`: A dictionary containing the killer's info.
        """
        # If the killer contains a link, it is a player.
        if "href" in killer:
            killer_dict = {"name": link_content.search(killer).group(1), "player": True}
        else:
            killer_dict = {"name": killer, "player": False}
        # Check if it contains a summon.
        m = death_summon.search(killer)
        if m:
            killer_dict["summon"] = m.group("summon")
        return killer_dict