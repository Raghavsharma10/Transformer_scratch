def check(self):
        """
        Checks values
        """
        status = True
        g = get_root(self).globals
        if self.mag.ok():
            self.mag.config(bg=g.COL['main'])
        else:
            self.mag.config(bg=g.COL['warn'])
            status = False

        if self.airmass.ok():
            self.airmass.config(bg=g.COL['main'])
        else:
            self.airmass.config(bg=g.COL['warn'])
            status = False

        if self.seeing.ok():
            self.seeing.config(bg=g.COL['main'])
        else:
            self.seeing.config(bg=g.COL['warn'])
            status = False

        return status