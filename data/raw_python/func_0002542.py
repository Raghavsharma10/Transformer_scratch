def _plan_on_valid_line(self, at_line, final_line_count):
        """Check if a plan is on a valid line."""
        # Put the common cases first.
        if at_line == 1 or at_line == final_line_count:
            return True

        # The plan may only appear on line 2 if the version is at line 1.
        after_version = (
            self._lines_seen["version"]
            and self._lines_seen["version"][0] == 1
            and at_line == 2
        )
        if after_version:
            return True

        return False