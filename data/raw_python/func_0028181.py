def lint_files(self, files):
        """ Lints a list of files.
        :param files: list of files to lint
        :return: a list of violations found in the files
        """
        all_violations = []
        for filename in files:
            with open(filename, 'r') as f:
                content = f.read()
                violations = self.lint(content)
                all_violations.extend(violations)
                for e in violations:
                    print("{0}:{1}: {2} {3}".format(filename, e.line_nr, e.rule_id, e.message))
        return len(all_violations)