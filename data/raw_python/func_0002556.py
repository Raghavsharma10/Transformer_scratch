def _load_lines(self, filename, line_generator, suite, rules):
        """Load a suite with lines produced by the line generator."""
        line_counter = 0
        for line in line_generator:
            line_counter += 1

            if line.category in self.ignored_lines:
                continue

            if line.category == "test":
                suite.addTest(Adapter(filename, line))
                rules.saw_test()
            elif line.category == "plan":
                if line.skip:
                    rules.handle_skipping_plan(line)
                    return suite
                rules.saw_plan(line, line_counter)
            elif line.category == "bail":
                rules.handle_bail(line)
                return suite
            elif line.category == "version":
                rules.saw_version_at(line_counter)

        rules.check(line_counter)
        return suite