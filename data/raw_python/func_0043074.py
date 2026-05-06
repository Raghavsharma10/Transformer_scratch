def list_from_tibiadata(cls, content):
        """Parses the content of a house list from TibiaData.com into a list of houses

        Parameters
        ----------
        content: :class:`str`
            The raw JSON response from TibiaData

        Returns
        -------
        :class:`list` of :class:`ListedHouse`

        Raises
        ------
        InvalidContent`
            Content is not the house list from TibiaData.com
        """
        json_data = parse_json(content)
        try:
            house_data = json_data["houses"]
            houses = []
            house_type = HouseType.HOUSE if house_data["type"] == "houses" else HouseType.GUILDHALL
            for house_json in house_data["houses"]:
                house = ListedHouse(house_json["name"], house_data["world"], house_json["houseid"],
                                    size=house_json["size"], rent=house_json["rent"], town=house_data["town"],
                                    type=house_type)
                house._parse_status(house_json["status"])
                houses.append(house)
            return houses
        except KeyError:
            raise InvalidContent("content is not a house list json response from TibiaData.com")