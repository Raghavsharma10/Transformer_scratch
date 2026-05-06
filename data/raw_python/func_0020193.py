def _render(self, contexts, partials):
        """render inverted section"""
        val = self._lookup(self.value, contexts)
        if val:
            return EMPTYSTRING
        return self._render_children(contexts, partials)