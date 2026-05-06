def parse_home_shots(self):
        """
        Parse shot info for home team.
        
        :returns: ``self`` on success, ``None`` otherwise
        """
        try:
            self.__set_shot_tables()
            self.shots['home'] = self.__parse_shot_tables(
                self.__home_top,
                self.__home_bot
            )
            return self
        except:
            return None