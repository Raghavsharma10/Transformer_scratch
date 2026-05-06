def set_break(self, break_type):
        """Set if the interpreter should break.

        :returns: TODO
        """
        self._break_type = break_type
        self._break_level = self._scope.level()