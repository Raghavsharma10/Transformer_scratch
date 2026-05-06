def from_tibiadata(cls, content):
        """Builds a character object from a TibiaData character response.

        Parameters
        ----------
        content: :class:`str`
            The JSON content of the response.

        Returns
        -------
        :class:`Character`
            The character contained in the page, or None if the character doesn't exist

        Raises
        ------
        InvalidContent
            If content is not a JSON string of the Character response."""
        json_content = parse_json(content)
        char = cls()
        try:
            character = json_content["characters"]
            if "error" in character:
                return None
            character_data = character["data"]
            char.name = character_data["name"]
            char.world = character_data["world"]
            char.level = character_data["level"]
            char.achievement_points = character_data["achievement_points"]
            char.sex = try_enum(Sex, character_data["sex"])
            char.vocation = try_enum(Vocation, character_data["vocation"])
            char.residence = character_data["residence"]
            char.account_status = try_enum(AccountStatus, character_data["account_status"])
        except KeyError:
            raise InvalidContent("content does not match a character json from TibiaData.")
        char.former_names = character_data.get("former_names", [])
        if "deleted" in character_data:
            char.deletion_date = parse_tibiadata_datetime(character_data["deleted"])
        char.married_to = character_data.get("married_to")
        char.former_world = character_data.get("former_world")
        char.position = character_data.get("Position:")
        if "guild" in character_data:
            char.guild_membership = GuildMembership(character_data["guild"]["name"], character_data["guild"]["rank"])
        if "house" in character_data:
            house = character_data["house"]
            paid_until_date = parse_tibiadata_date(house["paid"])
            char.house = CharacterHouse(house["houseid"], house["name"], char.world, house["town"], char.name,
                                        paid_until_date)
        char.comment = character_data.get("comment")
        if len(character_data["last_login"]) > 0:
            char.last_login = parse_tibiadata_datetime(character_data["last_login"][0])
        for achievement in character["achievements"]:
            char.achievements.append(Achievement(achievement["name"], achievement["stars"]))

        char._parse_deaths_tibiadata(character.get("deaths", []))

        for other_char in character["other_characters"]:
            char.other_characters.append(OtherCharacter(other_char["name"], other_char["world"],
                                                        other_char["status"] == "online",
                                                        other_char["status"] == "deleted"))

        if character["account_information"]:
            acc_info = character["account_information"]
            created = parse_tibiadata_datetime(acc_info.get("created"))
            loyalty_title = None if acc_info["loyalty_title"] == "(no title)" else acc_info["loyalty_title"]
            position = acc_info.get("position")

            char.account_information = AccountInformation(created, loyalty_title, position)

        return char