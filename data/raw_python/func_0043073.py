def list_from_content(cls, content):
        """Parses the content of a house list from Tibia.com into a list of houses

        Parameters
        ----------
        content: :class:`str`
            The raw HTML response from the house list.

        Returns
        -------
        :class:`list` of :class:`ListedHouse`

        Raises
        ------
        InvalidContent`
            Content is not the house list from Tibia.com
        """
        try:
            parsed_content = parse_tibiacom_content(content)
            table = parsed_content.find("table")
            header, *rows = table.find_all("tr")
        except (ValueError, AttributeError):
            raise InvalidContent("content does not belong to a Tibia.com house list")

        m = list_header_regex.match(header.text.strip())
        if not m:
            return None
        town = m.group("town")
        world = m.group("world")
        house_type = HouseType.GUILDHALL if m.group("type") == "Guildhalls" else HouseType.HOUSE
        houses = []
        for row in rows[1:]:
            cols = row.find_all("td")
            if len(cols) != 6:
                continue
            name = cols[0].text.replace('\u00a0', ' ')
            house = ListedHouse(name, world, 0, town=town, type=house_type)
            size = cols[1].text.replace('sqm', '')
            house.size = int(size)
            rent = cols[2].text.replace('gold', '')
            house.rent = int(rent)
            status = cols[3].text.replace('\xa0', ' ')
            house._parse_status(status)
            id_input = cols[5].find("input", {'name': 'houseid'})
            house.id = int(id_input["value"])
            houses.append(house)
        return houses