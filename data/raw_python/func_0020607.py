def parse_coaches(self):
        """
        Parse the home and away coaches

        :returns: ``self`` on success, ``None`` otherwise
        """
        lx_doc = self.html_doc()
        tr = lx_doc.xpath('//tr[@id="HeadCoaches"]')[0]

        for i, td in enumerate(tr):
            txt = td.xpath('.//text()')
            txt = ex_junk(txt, ['\n','\r'])
            team = 'away' if i == 0 else 'home'
            self.coaches[team] = txt[0]

        return self if self.coaches else None