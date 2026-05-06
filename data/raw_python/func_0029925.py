def rev(self, rev):
        """Return a new identity with the given revision"""
        d = self.dict
        d['revision'] = rev
        return self.from_dict(d)