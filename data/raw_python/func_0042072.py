def list_from_content(cls, content):
        """
        Gets a list of guilds from the HTML content of the world guilds' page.

        Parameters
        ----------
        content: :class:`str`
            The HTML content of the page.

        Returns
        -------
        :class:`list` of :class:`ListedGuild`
            List of guilds in the current world. ``None`` if it's the list of a world that doesn't exist.

        Raises
        ------
        InvalidContent
            If content is not the HTML of a guild's page.
        """
        parsed_content = parse_tibiacom_content(content)
        selected_world = parsed_content.find('option', selected=True)
        try:
            if "choose world" in selected_world.text:
                # It belongs to a world that doesn't exist
                return None
            world = selected_world.text
        except AttributeError:
            raise InvalidContent("Content does not belong to world guild list.")
        # First TableContainer contains world selector.
        _, *containers = parsed_content.find_all('div', class_="TableContainer")
        guilds = []
        for container in containers:
            header = container.find('div', class_="Text")
            active = "Active" in header.text
            header, *rows = container.find_all("tr", {'bgcolor': ["#D4C0A1", "#F1E0C6"]})
            for row in rows:
                columns = row.find_all('td')
                logo_img = columns[0].find('img')["src"]
                description_lines = columns[1].get_text("\n").split("\n", 1)
                name = description_lines[0]
                description = None
                if len(description_lines) > 1:
                    description = description_lines[1].replace("\r", "").replace("\n", " ")
                guild = cls(name, world, logo_img, description, active)
                guilds.append(guild)
        return guilds