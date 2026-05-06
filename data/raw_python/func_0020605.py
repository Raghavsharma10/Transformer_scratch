def parse_rosters(self):
        """
        Parse the home and away game rosters

        :returns: ``self`` on success, ``None`` otherwise
        """
        lx_doc = self.html_doc()

        if not self.__blocks:
            self.__pl_blocks(lx_doc)

        for t in ['home', 'away']:
            self.rosters[t] = self.__clean_pl_block(self.__blocks[t])

        return self if self.rosters else None