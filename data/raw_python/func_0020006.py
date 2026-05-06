def _init_draw(self):
        """Initializes the drawing of the frames by setting the images to
        random colors.

        This function is called by TimedAnimation.
        """
        if self.original is not None:
            self.original.set_data(np.random.random((10, 10, 3)))
        self.processed.set_data(np.random.random((10, 10, 3)))