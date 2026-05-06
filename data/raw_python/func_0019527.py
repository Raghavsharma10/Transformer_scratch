def best_match(self, seqs, scan_rc=True):
        """
        give the best match of each motif in each sequence
        returns an iterator of nested lists containing tuples:
        (score, position, strand)
        """
        self.set_threshold(threshold=0.0)
        for matches in self.scan(seqs, 1, scan_rc):
            yield [m[0] for m in matches]