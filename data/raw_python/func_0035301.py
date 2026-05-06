def Screens(self, text, prog, screen, clock):
        """Prog = 0 for first page, 1 for middle pages, 2 for last page"""
        # Initialize the screen class
        BaseScreen.__init__(self, self.size, self.background)

        # Determine the mid position of the given screen size and the
        # y button height
        xmid = self.size[0]//2

        # Create the header text
        Linesoftext(text, (xmid, 40), xmid=True, surface=self.image,
                    fontsize=30)

        # Create the buttons
        self.buttonlist = []
        if prog == 0:
            self.buttonlist += [self.nextbutton]

        elif prog == 1:
            self.buttonlist += [self.nextbutton]
            self.buttonlist += [self.backbutton]

        elif prog == 2:
            self.buttonlist += [self.lastbutton]
            self.buttonlist += [self.backbutton]

        # Draw the buttons to the screen
        for i in self.buttonlist:
            self.image.blit(*i.blitinfo)

        # Use the menu update method to run the screen and process button clicks
        return Menu.update(self, screen, clock)