def setExpertLevel(self):
        """
        Set expert level
        """
        g = get_root(self).globals
        level = g.cpars['expert_level']

        # now set whether buttons are permanently enabled or not
        if level == 0 or level == 1:
            self.load.setNonExpert()
            self.save.setNonExpert()
            self.unfreeze.setNonExpert()
            self.start.setNonExpert()
            self.stop.setNonExpert()

        elif level == 2:
            self.load.setExpert()
            self.save.setExpert()
            self.unfreeze.setExpert()
            self.start.setExpert()
            self.stop.setExpert()