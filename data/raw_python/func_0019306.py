def save_data(self, idx):
        """Call method `save_data|` of all handled |IOSequences|
        objects registered under |OutputSequencesABC|."""
        for subseqs in self:
            if isinstance(subseqs, abctools.OutputSequencesABC):
                subseqs.save_data(idx)