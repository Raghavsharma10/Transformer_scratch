def modver(self, *args):
        """
        Switches colour of verify button
        """
        g = get_root(self).globals
        if self.ok():
            tname = self.val.get()
            if tname in self.successes:
                # known to be in simbad
                self.verify.config(bg=g.COL['start'])
            elif tname in self.failures:
                # known not to be in simbad
                self.verify.config(bg=g.COL['stop'])
            else:
                # not known whether in simbad
                self.verify.config(bg=g.COL['main'])
            self.verify.config(state='normal')
        else:
            self.verify.config(bg=g.COL['main'])
            self.verify.config(state='disable')

        if self.callback is not None:
            self.callback()