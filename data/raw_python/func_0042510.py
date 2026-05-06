def _parse_achievements(self, rows):
        """
        Parses the character's displayed achievements

        Parameters
        ----------
        rows: :class:`list` of :class:`bs4.Tag`
            A list of all rows contained in the table.
        """
        for row in rows:
            cols = row.find_all('td')
            if len(cols) != 2:
                continue
            field, value = cols
            grade = str(field).count("achievement-grade-symbol")
            name = value.text.strip()
            self.achievements.append(Achievement(name, grade))