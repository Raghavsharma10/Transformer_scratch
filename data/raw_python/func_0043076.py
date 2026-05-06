def get_list_url_tibiadata(cls, world, town, house_type: HouseType = HouseType.HOUSE):
        """
        Gets the URL to the house list on Tibia.com with the specified parameters.

        Parameters
        ----------
        world: :class:`str`
            The name of the world.
        town: :class:`str`
            The name of the town.
        house_type: :class:`HouseType`
            Whether to search for houses or guildhalls.

        Returns
        -------
        :class:`str`
            The URL to the list matching the parameters.
        """
        house_type = "%ss" % house_type.value
        return HOUSE_LIST_URL_TIBIADATA % (urllib.parse.quote(world), urllib.parse.quote(town), house_type)