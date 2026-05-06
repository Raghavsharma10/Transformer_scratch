def get_world_list_url(cls, world):
        """Gets the Tibia.com URL for the guild section of a specific world.

        Parameters
        ----------
        world: :class:`str`
            The name of the world.

        Returns
        -------
        :class:`str`
            The URL to the guild's page
        """
        return GUILD_LIST_URL + urllib.parse.quote(world.title().encode('iso-8859-1'))