def _parse_battleye_status(self, battleye_string):
        """Parses the BattlEye string and applies the results.

        Parameters
        ----------
        battleye_string: :class:`str`
            String containing the world's Battleye Status.
        """
        m = battleye_regexp.search(battleye_string)
        if m:
            self.battleye_protected = True
            self.battleye_date = parse_tibia_full_date(m.group(1))
        else:
            self.battleye_protected = False
            self.battleye_date = None