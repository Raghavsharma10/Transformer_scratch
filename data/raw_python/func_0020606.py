def parse_scratches(self):
        """
        Parse the home and away healthy scratches

        :returns: ``self`` on success, ``None`` otherwise
        """
        lx_doc = self.html_doc()
        if not self.__blocks:
            self.__pl_blocks(lx_doc)

        for t in ['aw_scr', 'h_scr']:
            ix = 'away' if t == 'aw_scr' else 'home'
            self.scratches[ix] = self.__clean_pl_block(self.__blocks[t])

        return self if self.scratches else None