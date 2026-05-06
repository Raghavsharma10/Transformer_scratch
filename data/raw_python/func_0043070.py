def from_content(cls, content):
        """Parses a Tibia.com response into a House object.

        Parameters
        ----------
        content: :class:`str`
            HTML content of the page.

        Returns
        -------
        :class:`House`
            The house contained in the page, or None if the house doesn't exist.

        Raises
        ------
        InvalidContent
            If the content is not the house section on Tibia.com
        """
        parsed_content = parse_tibiacom_content(content)
        image_column, desc_column, *_ = parsed_content.find_all('td')
        if "Error" in image_column.text:
            return None
        image = image_column.find('img')
        for br in desc_column.find_all("br"):
            br.replace_with("\n")
        description = desc_column.text.replace("\u00a0", " ").replace("\n\n","\n")
        lines = description.splitlines()
        try:
            name, beds, info, state, *_ = lines
        except ValueError:
            raise InvalidContent("content does is not from the house section of Tibia.com")

        house = cls(name.strip())
        house.image_url = image["src"]
        house.id = int(id_regex.search(house.image_url).group(1))
        m = bed_regex.search(beds)
        if m:
            house.type = HouseType.GUILDHALL if m.group("type") in ["guildhall", "clanhall"] else HouseType.HOUSE
            beds_word = m.group("beds")
            if beds_word == "no":
                house.beds = 0
            else:
                house.beds = parse_number_words(beds_word)

        m = info_regex.search(info)
        if m:
            house.world = m.group("world")
            house.rent = int(m.group("rent"))
            house.size = int(m.group("size"))

        house._parse_status(state)
        return house