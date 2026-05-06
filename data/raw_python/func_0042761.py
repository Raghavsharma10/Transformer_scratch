def finish_hanging(self):
        """Add tokens for hanging singature if any"""
        if self.groups.starting_signature:
            if self.groups.starting_group:
                self.add_tokens_for_group(with_pass=True)

            elif self.groups.starting_single:
                self.add_tokens_for_single(ignore=True)