def sequence(self, per_exon=False):
        """
        Return the sequence for this feature.
        if per-exon is True, return an array of exon sequences
        This sequence is never reverse complemented
        """
        db = self.db
        if not per_exon:
            start = self.txStart + 1
            return _sequence(db, self.chrom, start, self.txEnd)
        else:
            # TODO: use same strategy as cds_sequence to reduce # of requests.
            seqs = []
            for start, end in self.exons:
                seqs.append(_sequence(db, self.chrom, start + 1, end))
            return seqs