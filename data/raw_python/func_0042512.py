def _parse_deaths(self, rows):
        """
        Parses the character's recent deaths

        Parameters
        ----------
        rows: :class:`list` of :class:`bs4.Tag`
            A list of all rows contained in the table.
        """
        for row in rows:
            cols = row.find_all('td')
            death_time_str = cols[0].text.replace("\xa0", " ").strip()
            death_time = parse_tibia_datetime(death_time_str)
            death = str(cols[1]).replace("\xa0", " ")
            death_info = death_regexp.search(death)
            if death_info:
                level = int(death_info.group("level"))
                killers_desc = death_info.group("killers")
            else:
                continue
            death = Death(self.name, level, time=death_time)
            assists_name_list = []
            # Check if the killers list contains assists
            assist_match = death_assisted.search(killers_desc)
            if assist_match:
                # Filter out assists
                killers_desc = assist_match.group("killers")
                # Split assists into a list.
                assists_name_list = self._split_list(assist_match.group("assists"))
            killers_name_list = self._split_list(killers_desc)
            for killer in killers_name_list:
                killer_dict = self._parse_killer(killer)
                death.killers.append(Killer(**killer_dict))
            for assist in assists_name_list:
                # Extract names from character links in assists list.
                assist_dict = {"name": link_content.search(assist).group(1), "player": True}
                death.assists.append(Killer(**assist_dict))
            try:
                self.deaths.append(death)
            except ValueError:
                # Some pvp deaths have no level, so they are raising a ValueError, they will be ignored for now.
                continue