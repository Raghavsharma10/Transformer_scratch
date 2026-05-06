def match_difficulty(self, value):
        """stub"""
        self._my_osid_query._add_match('texts.difficulty', str(value).lower(), True)