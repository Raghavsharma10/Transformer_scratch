def _parse_entry(self, cols):
        """Parses an entry's row and adds the result to py:attr:`entries`.

        Parameters
        ----------
        cols: :class:`bs4.ResultSet`
            The list of columns for that entry.
        """
        rank, name, vocation, *values = [c.text.replace('\xa0', ' ').strip() for c in cols]
        rank = int(rank)
        if self.category == Category.EXPERIENCE or self.category == Category.LOYALTY_POINTS:
            extra, value = values
        else:
            value, *extra = values
        value = int(value.replace(',', ''))
        if self.category == Category.EXPERIENCE:
            entry = ExpHighscoresEntry(name, rank, vocation, value, int(extra))
        elif self.category == Category.LOYALTY_POINTS:
            entry = LoyaltyHighscoresEntry(name, rank, vocation, value, extra)
        else:
            entry = HighscoresEntry(name, rank, vocation, value)
        self.entries.append(entry)