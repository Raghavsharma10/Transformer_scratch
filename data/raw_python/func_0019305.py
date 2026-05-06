def load_data(self, idx):
        """Call method |InputSequences.load_data| of all handled
        |InputSequences| objects."""
        for subseqs in self:
            if isinstance(subseqs, abctools.InputSequencesABC):
                subseqs.load_data(idx)