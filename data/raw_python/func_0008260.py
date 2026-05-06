def disable(self):
        """
        Disable the button, if in non-expert mode.
        """
        w.ActButton.disable(self)
        g = get_root(self).globals
        if self._expert:
            self.config(bg=g.COL['start'])
        else:
            self.config(bg=g.COL['startD'])