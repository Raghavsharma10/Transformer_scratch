def setExpertLevel(self):
        """
        Set expert level
        """
        g = get_root(self).globals
        level = g.cpars['expert_level']

        # first define which buttons are visible
        if level == 0:
            # simple layout
            for button in self.all_buttons:
                button.grid_forget()

            # then re-grid the two simple ones
            self.powerOn.grid(row=0, column=0)
            self.powerOff.grid(row=0, column=1)

        elif level == 1 or level == 2:
            # first remove all possible buttons
            for button in self.all_buttons:
                button.grid_forget()

            # restore detailed layout
            self.cldcOn.grid(row=0, column=1)
            self.cldcOff.grid(row=1, column=1)
            self.seqStart.grid(row=2, column=1)
            self.seqStop.grid(row=3, column=1)
            self.ngcOnline.grid(row=0, column=0)
            self.ngcOff.grid(row=1, column=0)
            self.ngcStandby.grid(row=2, column=0)
            self.ngcReset.grid(row=3, column=0)

        # now set whether buttons are permanently enabled or not
        if level == 0 or level == 1:
            for button in self.all_buttons:
                button.setNonExpert()

        elif level == 2:
            for button in self.all_buttons:
                button.setExpert()