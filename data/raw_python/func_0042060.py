def from_tibiadata(cls, content):
        """Builds a guild object from a TibiaData character response.

        Parameters
        ----------
        content: :class:`str`
            The json string from the TibiaData response.

        Returns
        -------
        :class:`Guild`
            The guild contained in the description or ``None``.

        Raises
        ------
        InvalidContent
            If content is not a JSON response of a guild's page.
        """
        json_content = parse_json(content)
        guild = cls()
        try:
            guild_obj = json_content["guild"]
            if "error" in guild_obj:
                return None
            guild_data = guild_obj["data"]
            guild.name = guild_data["name"]
            guild.world = guild_data["world"]
            guild.logo_url = guild_data["guildlogo"]
            guild.description = guild_data["description"]
            guild.founded = parse_tibiadata_date(guild_data["founded"])
            guild.open_applications = guild_data["application"]
        except KeyError:
            raise InvalidContent("content does not match a guild json from TibiaData.")
        guild.homepage = guild_data.get("homepage")
        guild.active = not guild_data.get("formation", False)
        if isinstance(guild_data["disbanded"], dict):
            guild.disband_date = parse_tibiadata_date(guild_data["disbanded"]["date"])
            guild.disband_condition = disband_tibadata_regex.search(guild_data["disbanded"]["notification"]).group(1)
        for rank in guild_obj["members"]:
            rank_name = rank["rank_title"]
            for member in rank["characters"]:
                guild.members.append(GuildMember(member["name"], rank_name, member["nick"] or None,
                                                 member["level"], member["vocation"],
                                                 joined=parse_tibiadata_date(member["joined"]),
                                                 online=member["status"] == "online"))
        for invited in guild_obj["invited"]:
            guild.invites.append(GuildInvite(invited["name"], parse_tibiadata_date(invited["invited"])))
        if isinstance(guild_data["guildhall"], dict):
            gh = guild_data["guildhall"]
            guild.guildhall = GuildHouse(gh["name"], gh["world"], guild.members[0].name,
                                         parse_tibiadata_date(gh["paid"]))
        return guild