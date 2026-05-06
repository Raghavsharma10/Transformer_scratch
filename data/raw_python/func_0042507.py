def from_content(cls, content):
        """Creates an instance of the class from the html content of the character's page.

        Parameters
        ----------
        content: :class:`str`
            The HTML content of the page.

        Returns
        -------
        :class:`Character`
            The character contained in the page, or None if the character doesn't exist

        Raises
        ------
        InvalidContent
            If content is not the HTML of a character's page.
        """
        parsed_content = parse_tibiacom_content(content)
        tables = cls._parse_tables(parsed_content)
        char = Character()
        if "Could not find character" in tables.keys():
            return None
        if "Character Information" in tables.keys():
            char._parse_character_information(tables["Character Information"])
        else:
            raise InvalidContent("content does not contain a tibia.com character information page.")
        char._parse_achievements(tables.get("Account Achievements", []))
        char._parse_deaths(tables.get("Character Deaths", []))
        char._parse_account_information(tables.get("Account Information", []))
        char._parse_other_characters(tables.get("Characters", []))
        return char