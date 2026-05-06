def home_shift_summ(self):
        """
        :returns: :py:class:`.ShiftSummary` by player for the home team
        :rtype: dict ``{ player_num: shift_summary_obj }``
        """
        if not self.__wrapped_home:
            self.__wrapped_home = self.__wrap(self._home.by_player)
        
        return self.__wrapped_home