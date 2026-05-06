def append_rules(self, an_iterable):
        """add rules to the TransformRuleSystem"""
        self.rules.extend(
            TransformRule(*x, config=self.config) for x in an_iterable
        )