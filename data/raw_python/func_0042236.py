def super_kls(self):
        """
            Determine what kls this group inherits from
            If default kls should be used, then None is returned
        """
        if not self.kls and self.parent and self.parent.name:
            return self.parent.kls_name
        return self.kls