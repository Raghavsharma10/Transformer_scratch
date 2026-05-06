def set_offset(self, offset, mid=None):
        """This method will allow the menu to be placed anywhere in the open
           window instead of just the upper left corner.
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Inputs:
            offset - This is the x,y tuple of the position that you want to
                move the screen to.

            mid - The offset will be treated as the value passed in instead of
                the top left pixel.

                'x' (the x point in offset will be treated as the middle of the
                      menu image)

                'y' (the y point in offset will be treated as the middle of the
                      menu image)

                'c' (the offset will be treated as the center of the menu image)

        (doc string updated ver 0.1)
        """

        ptg.BaseScreen.set_offset(self, offset, mid)
        for i in self.tilelist:
            for j in i:
                j.rect[0] += offset[0]
                j.rect[1] += offset[1]