def load_rules(self, an_iterable):
        """cycle through a collection of Transform rule tuples loading them
        into the TransformRuleSystem"""
        self.rules = [
            TransformRule(*x, config=self.config) for x in an_iterable
        ]