def run(self):
        """Required by flake8
        Will be called after add_options and parse_options.

        Yields:
            tuple: (int, int, str, type) the tuple used by flake8 to construct a violation
        """

        if len(self.filename_checks) == 0:
            message = "N401 no configuration found for {}, " \
                      "please provide filename configuration in a flake8 config".format(self.name)
            yield (0, 0, message, type(self))

        rule_funcs = [rules.rule_n5xx]

        for rule_func in rule_funcs:
            for rule_name, configured_rule in self.filename_checks.items():
                for err in rule_func(self.filename, rule_name, configured_rule, type(self)):
                    yield err