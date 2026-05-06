def parse_away_fo(self):
        """
        Parse face-off info for away team.
        
        :returns: ``self`` on success, ``None`` otherwise
        """
        try:
            self.__set_fo_tables()
            self.face_offs['away'] = self.__parse_fo_table(self.__away_fo)
            return self
        except:
            return None