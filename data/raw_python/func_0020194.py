def _render(self, contexts, partials):
        """render partials"""
        try:
            partial = partials[self.value]
        except KeyError as e:
            return self._escape(EMPTYSTRING)

        partial = re_insert_indent.sub(r'\1' + ' '*self.indent, partial)

        return inner_render(partial, contexts, partials, self.delimiter)