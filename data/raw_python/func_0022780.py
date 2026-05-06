def activate(self):
        """ Avoid overhead in calling glUseProgram with same arg.
        Warning: this will break if glUseProgram is used somewhere else.
        Per context we keep track of one current program.
        """
        if self._handle != self._parser.env.get('current_program', False):
            self._parser.env['current_program'] = self._handle
            gl.glUseProgram(self._handle)