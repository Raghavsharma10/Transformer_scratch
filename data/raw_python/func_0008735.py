def generate_backtrace(self, styles):
        """Return the (potentially) aligned, rebuit traceback

        Yes, we iterate over the entries thrice. We sacrifice
        performance for code readability. I mean.. come on, how long can
        your traceback be that it matters?
        """
        backtrace = []
        for entry in self.entries:
            backtrace.append(self.rebuild_entry(entry, styles))

        # Get the lenght of the longest string for each field of an entry
        lengths = self.align_all(backtrace) if self.align else [1, 1, 1, 1]

        aligned_backtrace = []
        for entry in backtrace:
            aligned_backtrace.append(self.align_entry(entry, lengths))
        return aligned_backtrace