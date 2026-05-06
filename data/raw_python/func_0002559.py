def generate_tap_reports(self):
        """Generate TAP reports.

        The results are either combined into a single output file or
        the output file name is generated from the test case.
        """
        # We're streaming but set_plan wasn't called, so we can only
        # know the plan now (at the end).
        if self.streaming and not self._plan_written:
            print("1..{0}".format(self.combined_line_number), file=self.stream)
            self._plan_written = True
            return

        if self.combined:
            combined_file = "testresults.tap"
            if self.outdir:
                combined_file = os.path.join(self.outdir, combined_file)
            with open(combined_file, "w") as out_file:
                self._write_tap_version(out_file)
                if self.plan is not None:
                    print("1..{0}".format(self.plan), file=out_file)
                for test_case in self.combined_test_cases_seen:
                    self.generate_tap_report(
                        test_case, self._test_cases[test_case], out_file
                    )
                if self.plan is None:
                    print("1..{0}".format(self.combined_line_number), file=out_file)
        else:
            for test_case, tap_lines in self._test_cases.items():
                with open(self._get_tap_file_path(test_case), "w") as out_file:
                    self._write_tap_version(out_file)
                    self.generate_tap_report(test_case, tap_lines, out_file)