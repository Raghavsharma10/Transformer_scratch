def from_content(cls, content):
        """Creates an instance of the class from the html content of a highscores page.

        Notes
        -----
        Tibia.com only shows up to 25 entries per page, so in order to obtain the full highscores, all 12 pages must
        be parsed and merged into one.

        Parameters
        ----------
        content: :class:`str`
            The HTML content of the page.

        Returns
        -------
        :class:`Highscores`
            The highscores results contained in the page.

        Raises
        ------
        InvalidContent
            If content is not the HTML of a highscore's page."""
        parsed_content = parse_tibiacom_content(content)
        tables = cls._parse_tables(parsed_content)
        filters = tables.get("Highscores Filter")
        if filters is None:
            raise InvalidContent("content does is not from the highscores section of Tibia.com")
        world_filter, vocation_filter, category_filter = filters
        world = world_filter.find("option", {"selected": True})["value"]
        if world == "":
            return None
        category = category_filter.find("option", {"selected": True})["value"]
        vocation_selected = vocation_filter.find("option", {"selected": True})
        vocation = int(vocation_selected["value"]) if vocation_selected else 0
        highscores = cls(world, category, vocation=vocation)
        entries = tables.get("Highscores")
        if entries is None:
            return None
        _, header, *rows = entries
        info_row = rows.pop()
        highscores.results_count = int(results_pattern.search(info_row.text).group(1))
        for row in rows:
            cols_raw = row.find_all('td')
            highscores._parse_entry(cols_raw)
        return highscores