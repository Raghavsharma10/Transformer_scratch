def _update(self):
        """Update the display of button after querying data from interface"""
        self.clear()
        self._set_boutons_communs()
        if self.interface:
            self.addSeparator()
            l_actions = self.interface.get_actions_toolbar()
            self._set_boutons_interface(l_actions)