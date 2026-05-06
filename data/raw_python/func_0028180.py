def _apply_line_rules(self, markdown_string):
        """ Iterates over the lines in a given markdown string and applies all the enabled line rules to each line """
        all_violations = []
        lines = markdown_string.split("\n")
        line_rules = self.line_rules
        line_nr = 1
        ignoring = False
        for line in lines:
            if ignoring:
                if line.strip() == '<!-- markdownlint:enable -->':
                    ignoring = False
            else:
                if line.strip() == '<!-- markdownlint:disable -->':
                    ignoring = True
                    continue

                for rule in line_rules:
                    violation = rule.validate(line)
                    if violation:
                        violation.line_nr = line_nr
                        all_violations.append(violation)
            line_nr += 1
        return all_violations