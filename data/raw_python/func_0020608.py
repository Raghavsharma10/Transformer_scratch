def parse_officials(self):
        """
        Parse the officials

        :returns: ``self`` on success, ``None`` otherwise
        """
        # begin proper body of method
        lx_doc = self.html_doc()
        off_parser = opm(self.game_key.season)
        self.officials = off_parser(lx_doc)

        return self if self.officials else None