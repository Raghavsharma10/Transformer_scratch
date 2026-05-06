def print_debug(self, text, indent=0):
        """Only prints debug info on screen when self.debug == True."""
        if self.debug:
            if indent > 0:
                print(" "*self.debug, text)
            self.debug += indent
            if indent <= 0:
                print(" "*self.debug, text)