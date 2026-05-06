def set_position(self, position, midpoint=False, surface=None):
        """This method allows the button to be moved manually and keep the click
        on functionality.
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Inputs:
            position - This is the x, y position of the top left corner of the
                            button. (defining point can be changed to midpoint)

            midpoint - If true is passed to midpoint the button will be blitted
                to a surface, either automatically if a surface is passed or
                manually, such that the position input is the center of the
                button rather than the top left corner.

        (doc string updated ver 0.1)"""

        # Find the image size and midpoint of the image
        imagesize = self.image.get_size()
        imagemidp = (int(imagesize[0] * 0.5), int(imagesize[1] * 0.5))

        # if a midpoint arguement is passed, set the pos to the top left pixel
        # such that the position passed in is in the middle of the button
        if midpoint:
            self.pos = (position[0] - imagemidp[0], position[1] - imagemidp[1])
        else:
            self.pos = position

        # set the rectangle to be used for collision detection
        self.rect = pygame.Rect(self.pos, self.image.get_size())

        # Set up the information that is needed to blit the image to the surface
        self.blitinfo = (self.image, self.pos)

        # automatically blit the button onto an input surface
        if surface:
            surface.blit(*self.blitinfo)