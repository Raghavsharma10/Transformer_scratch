def _parse_other_characters(self, rows):
        """
        Parses the character's other visible characters.

        Parameters
        ----------
        rows: :class:`list` of :class:`bs4.Tag`
            A list of all rows contained in the table.
        """
        for row in rows:
            cols_raw = row.find_all('td')
            cols = [ele.text.strip() for ele in cols_raw]
            if len(cols) != 5:
                continue
            name, world, status, __, __ = cols
            name = name.replace("\xa0", " ").split(". ")[1]
            self.other_characters.append(OtherCharacter(name, world, status == "online", status == "deleted"))