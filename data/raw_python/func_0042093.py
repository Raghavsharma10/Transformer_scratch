def from_content(cls, content):
        """Parses the content of the World Overview section from Tibia.com into an object of this class.

        Parameters
        ----------
        content: :class:`str`
            The HTML content of the World Overview page in Tibia.com

        Returns
        -------
        :class:`WorldOverview`
            An instance of this class containing all the information.

        Raises
        ------
        InvalidContent
            If the provided content is not the HTML content of the worlds section in Tibia.com
        """
        parsed_content = parse_tibiacom_content(content, html_class="TableContentAndRightShadow")
        world_overview = WorldOverview()
        try:
            record_row, titles_row, *rows = parsed_content.find_all("tr")
            m = record_regexp.search(record_row.text)
            if not m:
                raise InvalidContent("content does not belong to the World Overview section in Tibia.com")
            world_overview.record_count = int(m.group("count"))
            world_overview.record_date = parse_tibia_datetime(m.group("date"))
            world_rows = rows
            world_overview._parse_worlds(world_rows)
            return world_overview
        except (AttributeError, KeyError, ValueError):
            raise InvalidContent("content does not belong to the World Overview section in Tibia.com")