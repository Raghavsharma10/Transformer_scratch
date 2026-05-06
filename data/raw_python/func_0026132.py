def postcmd(self):
        """Make sure proper entry is activated when menu is posted"""
        value = self.choice.get()
        try:
            index = self.paramInfo.choice.index(value)
            self.entry.menu.activate(index)
        except ValueError:
            # initial null value may not be in list
            pass