def from_tibiadata(cls, content):
        """Parses a TibiaData.com response into a :class:`World`

        Parameters
        ----------
        content: :class:`str`
            The raw JSON content from TibiaData

        Returns
        -------
        :class:`World`
            The World described in the page, or ``None``.

        Raises
        ------
        InvalidContent
            If the provided content is not a TibiaData world response.
        """
        json_data = parse_json(content)
        try:
            world_data = json_data["world"]
            world_info = world_data["world_information"]
            world = cls(world_info["name"])
            if "location" not in world_info:
                return None
            world.online_count = world_info["players_online"]
            world.status = "Online" if world.online_count > 0 else "Offline"
            world.record_count = world_info["online_record"]["players"]
            world.record_date = parse_tibiadata_datetime(world_info["online_record"]["date"])
            world.creation_date = world_info["creation_date"]
            world.location = try_enum(WorldLocation, world_info["location"])
            world.pvp_type = try_enum(PvpType, world_info["pvp_type"])
            world.transfer_type = try_enum(TransferType, world_info.get("transfer_type"), TransferType.REGULAR)
            world.premium_only = "premium_type" in world_info
            world.world_quest_titles = world_info.get("world_quest_titles", [])
            world._parse_battleye_status(world_info.get("battleye_status", ""))
            world.experimental = world_info.get("Game World Type:", "Regular") != "Regular"
            for player in world_data.get("players_online", []):
                world.online_players.append(OnlineCharacter(player["name"], world.name, player["level"],
                                                            player["vocation"]))
            return world
        except KeyError:
            raise InvalidContent("content is not a world json response from TibiaData")