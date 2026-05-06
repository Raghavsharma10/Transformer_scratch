def draw(self, mode, selection):
        """ Draw program in given mode, with given selection (IndexBuffer or
        first, count).
        """
        if not self._linked:
            raise RuntimeError('Cannot draw program if code has not been set')
        # Init
        gl.check_error('Check before draw')
        mode = as_enum(mode)
        # Draw
        if len(selection) == 3:
            # Selection based on indices
            id_, gtype, count = selection
            if count:
                self._pre_draw()
                ibuf = self._parser.get_object(id_)
                ibuf.activate()
                gl.glDrawElements(mode, count, as_enum(gtype), None)
                ibuf.deactivate()
        else:
            # Selection based on start and count
            first, count = selection
            if count:
                self._pre_draw()
                gl.glDrawArrays(mode, first, count)
        # Wrap up
        gl.check_error('Check after draw')
        self._post_draw()