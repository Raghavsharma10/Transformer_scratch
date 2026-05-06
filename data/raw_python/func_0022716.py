def render(self):
        """ Render the canvas to an offscreen buffer and return the image
        array.

        Returns
        -------
        image : array
            Numpy array of type ubyte and shape (h, w, 4). Index [0, 0] is the 
            upper-left corner of the rendered region.
        
        """
        self.set_current()
        size = self.physical_size
        fbo = FrameBuffer(color=RenderBuffer(size[::-1]),
                          depth=RenderBuffer(size[::-1]))

        try:
            fbo.activate()
            self.events.draw()
            return fbo.read()
        finally:
            fbo.deactivate()