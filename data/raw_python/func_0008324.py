def setExpertLevel(self):
        """
        Modifies widget according to expertise level, which in this
        case is just matter of hiding or revealing the button to
        set CCD temps
        """
        g = get_root(self).globals
        level = g.cpars['expert_level']
        if level == 0:
            if self.val.get() == 'CCD TECs':
                self.val.set('Observe')
                self._changed()
            self.tecs.grid_forget()
        else:
            self.tecs.grid(row=0, column=3, sticky=tk.W)