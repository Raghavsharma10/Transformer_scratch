def toggle_shade(self, shade):
        """This method will overlay a semi-transparent shade on top of the
        tile's image.
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Inputs:
            shade - This will designate which shade you wish to turn on or off.
                Blue and red shades are available by default.

        (doc string updated ver 0.1)
        """

        # First toggle the user specified shade
        if self.shades[shade][0]:
            self.shades[shade][0] = 0
        else:
            self.shades[shade][0] = 1

        # Now draw the image with the active shades
        self.image.blit(self.pic, (0, 0))
        for key in self.shades:
            if self.shades[key][0]:
                self.image.blit(self.shades[key][1], (0, 0))