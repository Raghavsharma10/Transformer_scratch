def select_limit(self, source_expression):
        """Match LIMIT take [OFFSET drop]."""
        start = self.tokens.matched.start

        # The expression right after LIMIT is the count to take.
        limit_count_expression = self.expression()

        # Optional OFFSET follows.
        if self.tokens.accept(grammar.select_offset):
            offset_start = self.tokens.matched.start
            offset_end = self.tokens.matched.end

            # Next thing is the count to drop.
            offset_count_expression = self.expression()

            # We have a new source expression, which is drop(count, original).
            offset_source_expression = ast.Apply(
                ast.Var("drop", start=offset_start, end=offset_end,
                        source=self.original),
                offset_count_expression,
                source_expression,
                start=offset_start, end=offset_count_expression.end,
                source=self.original)

            # Drop before taking, because obviously.
            source_expression = offset_source_expression

        limit_expression = ast.Apply(
            ast.Var("take", start=start, end=limit_count_expression.end,
                    source=self.original),
            limit_count_expression,
            source_expression,
            start=start, end=self.tokens.matched.end, source=self.original)

        return limit_expression