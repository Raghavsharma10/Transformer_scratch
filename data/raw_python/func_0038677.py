def draw(self):
        """Draw all the sprites in the system using their renderers.

        This method is convenient to call from you Pyglet window's
        on_draw handler to redraw particles when needed.
        """
        glPushAttrib(GL_ALL_ATTRIB_BITS)
        self.draw_score()
        for sprite in self:
            sprite.draw()
        glPopAttrib()