def from_tibiadata(cls, content, vocation=None):
        """Builds a highscores object from a TibiaData highscores response.

        Notes
        -----
        Since TibiaData.com's response doesn't contain any indication of the vocation filter applied,
        :py:attr:`vocation` can't be determined from the response, so the attribute must be assigned manually.

        If the attribute is known, it can be passed for it to be assigned in this method.

        Parameters
        ----------
        content: :class:`str`
            The JSON content of the response.
        vocation: :class:`VocationFilter`, optional
            The vocation filter to assign to the results. Note that this won't affect the parsing.

        Returns
        -------
        :class:`Highscores`
            The highscores contained in the page, or None if the content is for the highscores of a nonexistent world.

        Raises
        ------
        InvalidContent
            If content is not a JSON string of the highscores response."""
        json_content = parse_json(content)
        try:
            highscores_json = json_content["highscores"]
            if "error" in highscores_json["data"]:
                return None
            world = highscores_json["world"]
            category = highscores_json["type"]
            highscores = cls(world, category)
            for entry in highscores_json["data"]:
                value_key = "level"
                if highscores.category in [Category.ACHIEVEMENTS, Category.LOYALTY_POINTS, Category.EXPERIENCE]:
                    value_key = "points"
                if highscores.category == Category.EXPERIENCE:
                    highscores.entries.append(ExpHighscoresEntry(entry["name"], entry["rank"], entry["voc"],
                                                                 entry[value_key], entry["level"]))
                elif highscores.category == Category.LOYALTY_POINTS:
                    highscores.entries.append(LoyaltyHighscoresEntry(entry["name"], entry["rank"], entry["voc"],
                                                                     entry[value_key], entry["title"]))
                else:
                    highscores.entries.append(HighscoresEntry(entry["name"], entry["rank"], entry["voc"],
                                                              entry[value_key]))
            highscores.results_count = len(highscores.entries)
        except KeyError:
            raise InvalidContent("content is not a TibiaData highscores response.")
        if isinstance(vocation, VocationFilter):
            highscores.vocation = vocation
        return highscores