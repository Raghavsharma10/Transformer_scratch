def draw(self, mode=None):
        """ Draw collection """

        if self._need_update:
            self._update()

        program = self._programs[0]

        mode = mode or self._mode
        if self._indices_list is not None:
            program.draw(mode, self._indices_buffer)
        else:
            program.draw(mode)