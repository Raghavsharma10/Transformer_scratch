def from_tibiadata(cls, content):
        """Parses the content of the World Overview section from TibiaData.com into an object of this class.

        Notes
        -----
        Due to TibiaData limitations, :py:attr:`record_count` and :py:attr:`record_date` are unavailable
        object.

        Additionally, the listed worlds in :py:attr:`worlds` lack some information when obtained from TibiaData.
        The following attributes are unavailable:

        - :py:attr:`ListedWorld.status` is always ``Online``.
        - :py:attr:`ListedWorld.battleye_protected` is always ``False``
        - :py:attr:`ListedWorld.battleye_date` is always ``None``.


        Parameters
        ----------
        content: :class:`str`
            The JSON response of the worlds section in TibiaData.com

        Returns
        -------
        :class:`WorldOverview`
            An instance of this class containing only the available worlds.

        Raises
        ------
        InvalidContent
            If the provided content is the json content of the world section in TibiaData.com
        """
        json_data = parse_json(content)
        try:
            worlds_json = json_data["worlds"]["allworlds"]
            world_overview = cls()
            for world_json in worlds_json:
                world = ListedWorld(world_json["name"], world_json["location"], world_json["worldtype"])
                world._parse_additional_info(world_json["additional"])
                world.online_count = world_json["online"]
                world_overview.worlds.append(world)
            return world_overview
        except KeyError:
            raise InvalidContent("content is not a worlds json response from TibiaData.com.")