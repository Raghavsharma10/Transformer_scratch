def from_content(cls, content):
        """Creates an instance of the class from the HTML content of the guild's page.

        Parameters
        -----------
        content: :class:`str`
            The HTML content of the page.

        Returns
        ----------
        :class:`Guild`
            The guild contained in the page or None if it doesn't exist.

        Raises
        ------
        InvalidContent
            If content is not the HTML of a guild's page.
        """
        if "An internal error has occurred" in content:
            return None

        parsed_content = parse_tibiacom_content(content)
        try:
            name_header = parsed_content.find('h1')
            guild = Guild(name_header.text.strip())
        except AttributeError:
            raise InvalidContent("content does not belong to a Tibia.com guild page.")

        if not guild._parse_logo(parsed_content):
            raise InvalidContent("content does not belong to a Tibia.com guild page.")

        info_container = parsed_content.find("div", id="GuildInformationContainer")
        guild._parse_guild_info(info_container)
        guild._parse_application_info(info_container)
        guild._parse_guild_homepage(info_container)
        guild._parse_guild_guildhall(info_container)
        guild._parse_guild_disband_info(info_container)
        guild._parse_guild_members(parsed_content)

        if guild.guildhall and guild.members:
            guild.guildhall.owner = guild.members[0].name

        return guild