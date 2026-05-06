def _get_hook(self, shader, name):
        """Return a FunctionChain that Filters may use to modify the program.

        *shader* should be "frag" or "vert"
        *name* should be "pre" or "post"
        """
        assert name in ('pre', 'post')
        key = (shader, name)
        if key in self._hooks:
            return self._hooks[key]
        hook = StatementList()
        if shader == 'vert':
            self.view_program.vert[name] = hook
        elif shader == 'frag':
            self.view_program.frag[name] = hook
        self._hooks[key] = hook
        return hook