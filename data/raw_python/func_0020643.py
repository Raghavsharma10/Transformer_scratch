def parse_away_shots(self):
        """
        Parse shot info for away team.
        
        :returns: ``self`` on success, ``None`` otherwise
        """
        try:
            self.__set_shot_tables()
            self.shots['away'] = self.__parse_shot_tables(
                self.__aw_top,
                self.__aw_bot
            )
            return self
        except:
            return None