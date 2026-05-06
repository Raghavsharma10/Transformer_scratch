def _render(self, contexts, partials):
        """render variable"""
        value = self._lookup(self.value, contexts)

        # lambda
        if callable(value):
            value = inner_render(str(value()), contexts, partials)

        return self._escape(value)