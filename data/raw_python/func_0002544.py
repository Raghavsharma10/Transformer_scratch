def handle_skipping_plan(self, skip_plan):
        """Handle a plan that contains a SKIP directive."""
        skip_line = Result(True, None, skip_plan.directive.text, Directive("SKIP"))
        self._suite.addTest(Adapter(self._filename, skip_line))