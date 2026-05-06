def _render_children(self, contexts, partials):
        """Render the children tokens"""
        ret = []
        for child in self.children:
            ret.append(child._render(contexts, partials))
        return EMPTYSTRING.join(ret)