def _parse_plan(self, match):
        """Parse a matching plan line."""
        expected_tests = int(match.group("expected"))
        directive = Directive(match.group("directive"))

        # Only SKIP directives are allowed in the plan.
        if directive.text and not directive.skip:
            return Unknown()

        return Plan(expected_tests, directive)