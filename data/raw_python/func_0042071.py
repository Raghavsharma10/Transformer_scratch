def get_world_list_url_tibiadata(cls, world):
        """Gets the TibiaData.com URL for the guild list of a specific world.

        Parameters
        ----------
        world: :class:`str`
            The name of the world.

        Returns
        -------
        :class:`str`
            The URL to the guild's page.
        """
        return GUILD_LIST_URL_TIBIADATA % urllib.parse.quote(world.title().encode('iso-8859-1'))