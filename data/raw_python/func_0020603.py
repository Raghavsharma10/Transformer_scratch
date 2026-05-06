def away_shift_summ(self):
        """
        :returns: :py:class:`.ShiftSummary` by player for the away team
        :rtype: dict ``{ player_num: shift_summary_obj }``
        """
        if not self.__wrapped_away:
            self.__wrapped_away = self.__wrap(self._away.by_player)
        
        return self.__wrapped_away