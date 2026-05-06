def to_dict(self):
        """Convert back to the pstats dictionary representation (used for saving back as pstats binary file)"""
        if self.subcall is not None:
            if isinstance(self.subcall, dict):
                subcalls = self.subcall
            else:
                subcalls = {}
                for s in self.subcall:
                    subcalls.update(s.to_dict())
            return {(self.filename, self.line_number, self.name): \
                        (self.ncalls, self.nonrecursive_calls, self.own_time_s, self.cummulative_time_s, subcalls)}
        else:
            return {(self.filename, self.line_number, self.name): \
                        (self.ncalls, self.nonrecursive_calls, self.own_time_s, self.cummulative_time_s)}