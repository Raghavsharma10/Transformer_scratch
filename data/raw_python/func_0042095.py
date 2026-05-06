def _parse_worlds(self, world_rows):
        """Parses the world columns and adds the results to :py:attr:`worlds`.

        Parameters
        ----------
        world_rows: :class:`list` of :class:`bs4.Tag`
            A list containing the rows of each world.
        """
        for world_row in world_rows:
            cols = world_row.find_all("td")
            name = cols[0].text.strip()
            status = "Online"
            try:
                online = int(cols[1].text.strip())
            except ValueError:
                online = 0
                status = "Offline"
            location = cols[2].text.replace("\u00a0", " ").strip()
            pvp = cols[3].text.strip()

            world = ListedWorld(name, location, pvp, online_count=online, status=status)
            # Check Battleye icon to get information
            battleye_icon = cols[4].find("span", attrs={"class": "HelperDivIndicator"})
            if battleye_icon is not None:
                world.battleye_protected = True
                m = battleye_regexp.search(battleye_icon["onmouseover"])
                if m:
                    world.battleye_date = parse_tibia_full_date(m.group(1))

            additional_info = cols[5].text.strip()
            world._parse_additional_info(additional_info)
            self.worlds.append(world)