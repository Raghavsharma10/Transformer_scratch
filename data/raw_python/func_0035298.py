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

        if mid:
            imagesize = self.image.get_size()
            imagemidp = (int(imagesize[0] * 0.5), int(imagesize[1] * 0.5))
            if mid == 'x':
                offset = (offset[0] - imagemidp[0], offset[1])
            if mid == 'y':
                offset = (offset[0], offset[1] - imagemidp[1])
            if mid == 'c':
                offset = (offset[0] - imagemidp[0], offset[1] - imagemidp[1])

        self.pos = offset

        for i in self.buttonlist:
                i.rect[0] += offset[0]
                i.rect[1] += offset[1]

        try:
            for i in self.widgetlist:
                i.rect[0] += offset[0]
                i.rect[1] += offset[1]
        except AttributeError:
            pass