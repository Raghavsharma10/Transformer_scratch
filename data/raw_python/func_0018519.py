def blat(self, db=None, sequence=None, seq_type="DNA"):
        """
        make a request to the genome-browsers BLAT interface
        sequence is one of None, "mrna", "cds"
        returns a list of features that are hits to this sequence.
        """
        from . blat_blast import blat, blat_all
        assert sequence in (None, "cds", "mrna")
        seq = self.sequence() if sequence is None else ("".join(self.cds_sequence if sequence == "cds" else self.mrna_sequence))
        if isinstance(db, (tuple, list)):
            return blat_all(seq, self.gene_name, db, seq_type)
        else:
            return blat(seq, self.gene_name, db or self.db, seq_type)