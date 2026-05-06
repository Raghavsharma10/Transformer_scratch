def _parse_guild_members(self, parsed_content):
        """
        Parses the guild's member and invited list.

        Parameters
        ----------
        parsed_content: :class:`bs4.Tag`
            The parsed content of the guild's page
        """
        member_rows = parsed_content.find_all("tr", {'bgcolor': ["#D4C0A1", "#F1E0C6"]})
        previous_rank = {}
        for row in member_rows:
            columns = row.find_all('td')
            values = tuple(c.text.replace("\u00a0", " ") for c in columns)
            if len(columns) == COLS_GUILD_MEMBER:
                self._parse_current_member(previous_rank, values)
            if len(columns) == COLS_INVITED_MEMBER:
                self._parse_invited_member(values)