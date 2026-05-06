def _track(self, class_name):
        """Keep track of which test cases have executed."""
        if self._test_cases.get(class_name) is None:
            if self.streaming and self.header:
                self._write_test_case_header(class_name, self.stream)

            self._test_cases[class_name] = []
            if self.combined:
                self.combined_test_cases_seen.append(class_name)