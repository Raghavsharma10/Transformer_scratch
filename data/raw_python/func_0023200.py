def build_if_needed(self):
        """ Reset shader source if necesssary.
        """
        if self._need_build:
            self._build()
            self._need_build = False
        self.update_variables()