def get_url(cls, world, category=Category.EXPERIENCE, vocation=VocationFilter.ALL, page=1):
        """Gets the Tibia.com URL of the highscores for the given parameters.

        Parameters
        ----------
        world: :class:`str`
            The game world of the desired highscores.
        category: :class:`Category`
            The desired highscores category.
        vocation: :class:`VocationFiler`
            The vocation filter to apply. By default all vocations will be shown.
        page: :class:`int`
            The page of highscores to show.

        Returns
        -------
        The URL to the Tibia.com highscores.
        """
        return HIGHSCORES_URL % (world, category.value, vocation.value, page)