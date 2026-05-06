def add_screen(self, ref):

        """ Add Screen """

        if ref not in self.screens:
            screen = Screen(self, ref)
            screen.clear()              # TODO Check this is needed, new screens should be clear.
            self.screens[ref] = screen
            return self.screens[ref]