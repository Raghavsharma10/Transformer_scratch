def match(self, version):
        """Check whether a Version satisfies the Spec."""
        return all(spec.match(version) for spec in self.specs)