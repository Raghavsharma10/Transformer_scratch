def visit_Call(self, node):
        """Call visitor - used for finding setup() call."""
        self.generic_visit(node)

        # Setup() is a keywords-only function.
        if node.args:
            return

        keywords = set()
        for k in node.keywords:
            if k.arg is not None:
                keywords.add(k.arg)
            # Simple case for dictionary expansion for Python >= 3.5.
            if k.arg is None and isinstance(k.value, ast.Dict):
                keywords.update(x.s for x in k.value.keys)
        # Simple case for dictionary expansion for Python <= 3.4.
        if getattr(node, 'kwargs', ()) and isinstance(node.kwargs, ast.Dict):
            keywords.update(x.s for x in node.kwargs.keys)

        # The bare minimum number of arguments seems to be around five, which
        # includes author, name, version, module/package and something extra.
        if len(keywords) < 5:
            return

        score = sum(
            self.attributes.get(x, 0)
            for x in keywords
        ) / len(keywords)

        if score < 0.5:
            LOG.debug(
                "Scoring for setup%r below 0.5: %.2f",
                tuple(keywords),
                score)
            return

        # Redirect call to our setup() tap function.
        node.func = ast.Name(id='__f8r_setup', ctx=node.func.ctx)
        self.redirected = True