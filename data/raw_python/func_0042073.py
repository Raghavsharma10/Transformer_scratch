def list_from_tibiadata(cls, content):
        """Builds a character object from a TibiaData character response.

        Parameters
        ----------
        content: :class:`str`
            A string containing the JSON response from TibiaData.

        Returns
        -------
        :class:`list` of :class:`ListedGuild`
            The list of guilds contained.

        Raises
        ------
        InvalidContent
            If content is not a JSON response of TibiaData's guild list.
        """
        json_content = parse_json(content)
        try:
            guilds_obj = json_content["guilds"]
            guilds = []
            for guild in guilds_obj["active"]:
                guilds.append(cls(guild["name"], guilds_obj["world"], logo_url=guild["guildlogo"],
                                  description=guild["desc"], active=True))
            for guild in guilds_obj["formation"]:
                guilds.append(cls(guild["name"], guilds_obj["world"], logo_url=guild["guildlogo"],
                                  description=guild["desc"], active=False))
        except KeyError:
            raise InvalidContent("content doest not belong to a guilds response.")
        return guilds