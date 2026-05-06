def _process_plan_lines(self, final_line_count):
        """Process plan line rules."""
        if not self._lines_seen["plan"]:
            self._add_error(_("Missing a plan."))
            return

        if len(self._lines_seen["plan"]) > 1:
            self._add_error(_("Only one plan line is permitted per file."))
            return

        plan, at_line = self._lines_seen["plan"][0]
        if not self._plan_on_valid_line(at_line, final_line_count):
            self._add_error(
                _("A plan must appear at the beginning or end of the file.")
            )
            return

        if plan.expected_tests != self._lines_seen["test"]:
            self._add_error(
                _("Expected {expected_count} tests but only {seen_count} ran.").format(
                    expected_count=plan.expected_tests,
                    seen_count=self._lines_seen["test"],
                )
            )