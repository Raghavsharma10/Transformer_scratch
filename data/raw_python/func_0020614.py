def parse_matchup(self):
        """
        Parse the banner matchup meta info for the game.
        
        :returns: ``self`` on success or ``None``
        """
        lx_doc = self.html_doc()
        try:
            if not self.matchup:
                self.matchup = self._fill_meta(lx_doc)
            return self
        except:
            return None