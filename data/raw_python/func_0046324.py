def _append(self, sh):
        '''
        Internal. Chains a command after this.
        :param sh: Next command.
        '''
        sh._input = self
        self._output = sh
        if self._env:
            sh._env = dict(self._env)
        if self._cwd:
            sh._cwd = self._cwd