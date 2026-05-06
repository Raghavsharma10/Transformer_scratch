def duration(self):
        """Get duration of composition
        """
        return max([x.comp_location + x.duration
                    for x in self.segments])