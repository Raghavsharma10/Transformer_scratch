def from_tibiadata(cls, content):
        """
        Parses a TibiaData response into a House object.

        Parameters
        ----------
        content: :class:`str`
            The JSON content of the TibiaData response.

        Returns
        -------
        :class:`House`
            The house contained in the response, if found.

        Raises
        ------
        InvalidContent
            If the content is not a house JSON response from TibiaData
        """
        json_content = parse_json(content)
        try:
            house_json = json_content["house"]
            if not house_json["name"]:
                return None
            house = cls(house_json["name"], house_json["world"])

            house.type = try_enum(HouseType, house_json["type"])
            house.id = house_json["houseid"]
            house.beds = house_json["beds"]
            house.size = house_json["size"]
            house.size = house_json["size"]
            house.rent = house_json["rent"]
            house.image_url = house_json["img"]

            # Parsing the original status string is easier than dealing with TibiaData fields
            house._parse_status(house_json["status"]["original"])
        except KeyError:
            raise InvalidContent("content is not a TibiaData house response.")
        return house