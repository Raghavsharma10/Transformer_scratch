def _render(self, contexts, partials):
        """render section"""
        val = self._lookup(self.value, contexts)
        if not val:
            # false value
            return EMPTYSTRING

        # normally json has types: number/string/list/map
        # but python has more, so we decide that map and string should not iterate
        # by default, other do.
        if hasattr(val, "__iter__") and not isinstance(val, (str, dict)):
            # non-empty lists
            ret = []

            for item in val:
                contexts.append(item)
                ret.append(self._render_children(contexts, partials))
                contexts.pop()

            if len(ret) <= 0:
                # empty lists
                return EMPTYSTRING

            return self._escape(''.join(ret))
        elif callable(val):
            # lambdas
            new_template = val(self.text)
            value = inner_render(new_template, contexts, partials, self.delimiter)
        else:
            # context
            contexts.append(val)
            value = self._render_children(contexts, partials)
            contexts.pop()

        return self._escape(value)