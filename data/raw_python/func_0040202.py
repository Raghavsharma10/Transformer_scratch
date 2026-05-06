def is_discrete(self):
        """
        Return whether this space is discrete
        """
        for domain in self.domains.values():
            if not domain.is_discrete():
                return False
        return True