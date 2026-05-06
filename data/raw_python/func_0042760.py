def add_tokens_for_single(self, ignore=False):
        """Add the tokens for the single signature"""
        args = self.single.args
        name = self.single.python_name

        # Reset indentation to proper amount and add signature
        self.reset_indentation(self.indent_type * self.single.indent)
        self.result.extend(self.tokens.make_single(name, args))

        # Add skip if necessary
        if ignore:
            self.single.skipped = True
            self.result.extend(self.tokens.test_skip)

        self.groups.finish_signature()