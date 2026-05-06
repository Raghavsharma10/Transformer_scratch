def from_content(cls, content):
        """Parses a Tibia.com response into a :class:`World`.

        Parameters
        ----------
        content: :class:`str`
            The raw HTML from the server's information page.

        Returns
        -------
        :class:`World`
            The World described in the page, or ``None``.

        Raises
        ------
        InvalidContent
            If the provided content is not the html content of the world section in Tibia.com
        """
        parsed_content = parse_tibiacom_content(content)
        tables = cls._parse_tables(parsed_content)
        try:
            error = tables.get("Error")
            if error and error[0].text == "World with this name doesn't exist!":
                return None
            selected_world = parsed_content.find('option', selected=True)
            world = cls(selected_world.text)
            world._parse_world_info(tables.get("World Information", []))

            online_table = tables.get("Players Online", [])
            world.online_players = []
            for row in online_table[1:]:
                cols_raw = row.find_all('td')
                name, level, vocation = (c.text.replace('\xa0', ' ').strip() for c in cols_raw)
                world.online_players.append(OnlineCharacter(name, world.name, int(level), vocation))
        except AttributeError:
            raise InvalidContent("content is not from the world section in Tibia.com")

        return world