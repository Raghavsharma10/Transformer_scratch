def add_tokens_for_group(self, with_pass=False):
        """Add the tokens for the group signature"""
        kls = self.groups.super_kls
        name = self.groups.kls_name

        # Reset indentation to beginning and add signature
        self.reset_indentation('')
        self.result.extend(self.tokens.make_describe(kls, name))

        # Add pass if necessary
        if with_pass:
            self.add_tokens_for_pass()

        self.groups.finish_signature()