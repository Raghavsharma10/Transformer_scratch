def post_from_style(self, column_style):
        """A Terminal-specific reset to StyleProcessors.post_from_style.
        """
        for proc in super(TermProcessors, self).post_from_style(column_style):
            if proc.__name__ == "join_flanks":
                # Reset any codes before adding back whitespace.
                yield self._maybe_reset()
            yield proc