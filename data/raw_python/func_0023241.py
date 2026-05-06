def _prepare_draw(self, view=None):
        """This method is called immediately before each draw.

        The *view* argument indicates which view is about to be drawn.
        """

        if self._changed['pos']:
            self.pos_buf.set_data(self._pos)
            self._changed['pos'] = False

        if self._changed['color']:
            self.color_buf.set_data(self._color)
            self._program.vert['color'] = self.color_buf
            self._changed['color'] = False

        return True