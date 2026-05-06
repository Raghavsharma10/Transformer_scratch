def _parse_character_information(self, rows):
        """
        Parses the character's basic information and applies the found values.

        Parameters
        ----------
        rows: :class:`list` of :class:`bs4.Tag`
            A list of all rows contained in the table.
        """
        int_rows = ["level", "achievement_points"]
        char = {}
        house = {}
        for row in rows:
            cols_raw = row.find_all('td')
            cols = [ele.text.strip() for ele in cols_raw]
            field, value = cols
            field = field.replace("\xa0", "_").replace(" ", "_").replace(":", "").lower()
            value = value.replace("\xa0", " ")
            # This is a special case cause we need to see the link
            if field == "house":
                house_text = value
                paid_until = house_regexp.search(house_text).group(1)
                paid_until_date = parse_tibia_date(paid_until)
                house_link = cols_raw[1].find('a')
                url = urllib.parse.urlparse(house_link["href"])
                query = urllib.parse.parse_qs(url.query)
                house = {"id": int(query["houseid"][0]), "name": house_link.text.strip(),
                         "town": query["town"][0], "paid_until": paid_until_date}
                continue
            if field in int_rows:
                value = int(value)
            char[field] = value

        # If the character is deleted, the information is fouund with the name, so we must clean it
        m = deleted_regexp.match(char["name"])
        if m:
            char["name"] = m.group(1)
            char["deletion_date"] = parse_tibia_datetime(m.group(2))
        if "guild_membership" in char:
            m = guild_regexp.match(char["guild_membership"])
            char["guild_membership"] = GuildMembership(m.group(2), m.group(1))

        if "former_names" in char:
            former_names = [fn.strip() for fn in char["former_names"].split(",")]
            char["former_names"] = former_names

        if "never" in char["last_login"]:
            char["last_login"] = None
        else:
            char["last_login"] = parse_tibia_datetime(char["last_login"])

        char["vocation"] = try_enum(Vocation, char["vocation"])
        char["sex"] = try_enum(Sex, char["sex"])
        char["account_status"] = try_enum(AccountStatus, char["account_status"])

        for k, v in char.items():
            try:
                setattr(self, k, v)
            except AttributeError:
                pass
        if house:
            self.house = CharacterHouse(house["id"], house["name"], self.world, house["town"], self.name,
                                        house["paid_until"])