def _parse_world_info(self, world_info_table):
        """
        Parses the World Information table from Tibia.com and adds the found values to the object.

        Parameters
        ----------
        world_info_table: :class:`list`[:class:`bs4.Tag`]
        """
        world_info = {}
        for row in world_info_table:
            cols_raw = row.find_all('td')
            cols = [ele.text.strip() for ele in cols_raw]
            field, value = cols
            field = field.replace("\xa0", "_").replace(" ", "_").replace(":", "").lower()
            value = value.replace("\xa0", " ")
            world_info[field] = value
        try:
            self.online_count = int(world_info.pop("players_online"))
        except KeyError:
            self.online_count = 0
        self.location = try_enum(WorldLocation, world_info.pop("location"))
        self.pvp_type = try_enum(PvpType, world_info.pop("pvp_type"))
        self.transfer_type = try_enum(TransferType, world_info.pop("transfer_type", None), TransferType.REGULAR)
        m = record_regexp.match(world_info.pop("online_record"))
        if m:
            self.record_count = int(m.group("count"))
            self.record_date = parse_tibia_datetime(m.group("date"))
        if "world_quest_titles" in world_info:
            self.world_quest_titles = [q.strip() for q in world_info.pop("world_quest_titles").split(",")]
        self.experimental = world_info.pop("game_world_type") != "Regular"
        self._parse_battleye_status(world_info.pop("battleye_status"))
        self.premium_only = "premium_type" in world_info

        month, year = world_info.pop("creation_date").split("/")
        month = int(month)
        year = int(year)
        if year > 90:
            year += 1900
        else:
            year += 2000
        self.creation_date = "%d-%02d" % (year, month)

        for k, v in world_info.items():
            try:
                setattr(self, k, v)
            except AttributeError:
                pass