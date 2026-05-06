def _create_output_from_match(self, match_result):
        """Create Result instance from pattern match results.

        Args:
            match: Pattern match.
        """
        if isinstance(match_result, dict):
            return LinterOutput(self.name, **match_result)
        return LinterOutput(self.name, *match_result)