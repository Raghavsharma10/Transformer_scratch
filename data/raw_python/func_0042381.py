def get_url_tibiadata(cls, world, category=Category.EXPERIENCE, vocation=VocationFilter.ALL):
        """Gets the TibiaData.com URL of the highscores for the given parameters.

        Parameters
        ----------
        world: :class:`str`
            The game world of the desired highscores.
        category: :class:`Category`
            The desired highscores category.
        vocation: :class:`VocationFiler`
            The vocation filter to apply. By default all vocations will be shown.

        Returns
        -------
        The URL to the TibiaData.com highscores.
        """
        return HIGHSCORES_URL_TIBIADATA % (world, category.value.lower(), vocation.name.lower())