def freeze(self):
        """
        Freeze all settings so that they can't be altered
        """
        self.target.disable()
        self.filter.configure(state='disable')
        self.prog_ob.configure(state='disable')
        self.pi.configure(state='disable')
        self.observers.configure(state='disable')
        self.comment.configure(state='disable')