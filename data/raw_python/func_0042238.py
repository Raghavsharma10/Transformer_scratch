def start_single(self, typ, scol):
        """Start a new single"""
        self.starting_single = True
        single = self.single = Single(typ=typ, group=self, indent=(scol - self.level))
        self.singles.append(single)
        return single