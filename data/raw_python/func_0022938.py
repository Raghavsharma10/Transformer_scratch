def activate(self):
        """ Activate/use this frame buffer.
        """
        # Send command
        self._glir.command('FRAMEBUFFER', self._id, True)
        # Associate canvas now
        canvas = get_current_canvas()
        if canvas is not None:
            canvas.context.glir.associate(self.glir)