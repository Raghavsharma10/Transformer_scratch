def parse_home_fo(self):
        """
        Parse face-off info for home team.
        
        :returns: ``self`` on success, ``None`` otherwise
        """
        try:
            self.__set_fo_tables()
            self.face_offs['home'] = self.__parse_fo_table(self.__home_fo)
            return self
        except:
            return None