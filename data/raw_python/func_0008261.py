def setExpert(self):
        """
        Turns on 'expert' status whereby the button is always enabled,
        regardless of its activity status.
        """
        w.ActButton.setExpert(self)
        g = get_root(self).globals
        self.config(bg=g.COL['start'])