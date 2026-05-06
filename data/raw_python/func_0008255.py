def unfreeze(self):
        """
        Unfreeze all settings so that they can be altered
        """
        g = get_root(self).globals
        self.filter.configure(state='normal')
        dtype = g.observe.rtype()
        if dtype == 'data caution' or dtype == 'data' or dtype == 'technical':
            self.prog_ob.configure(state='normal')
            self.pi.configure(state='normal')
            self.target.enable()
        self.observers.configure(state='normal')
        self.comment.configure(state='normal')