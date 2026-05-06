def translate_path(self, dep_file, dep_rule):
        """Translate dep_file from dep_rule into this rule's output path."""
        dst_base = dep_file.split(os.path.join(dep_rule.address.repo,
                                               dep_rule.address.path), 1)[-1]
        if self.params['strip_prefix']:
            dst_base = dep_file.split(self.params['strip_prefix'], 1)[-1]
        return os.path.join(self.address.repo, self.address.path,
                            self.params['prefix'].lstrip('/'),
                            dst_base.lstrip('/'))