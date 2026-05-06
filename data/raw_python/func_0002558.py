def set_plan(self, total):
        """Notify the tracker how many total tests there will be."""
        self.plan = total
        if self.streaming:
            # This will only write the plan if we haven't written it
            # already but we want to check if we already wrote a
            # test out (in which case we can't just write the plan out
            # right here).
            if not self.combined_test_cases_seen:
                self._write_plan(self.stream)
        elif not self.combined:
            raise ValueError(
                "set_plan can only be used with combined or streaming output"
            )