def features(self, other_start, other_end):
        """
        return e.g. "intron;exon" if the other_start, end overlap introns and
        exons
        """
        # completely encases gene.
        if other_start <= self.start and other_end >= self.end:
            return ['gene' if self.cdsStart != self.cdsEnd else 'nc_gene']
        other = Interval(other_start, other_end)
        ovls = []
        tx = 'txEnd' if self.strand == "-" else 'txStart'
        if hasattr(self, tx) and other_start <= getattr(self, tx) <= other_end \
            and self.cdsStart != self.cdsEnd:
                ovls = ["TSS"]
        for ftype in ('introns', 'exons', 'utr5', 'utr3', 'cdss'):
            feats = getattr(self, ftype)
            if not isinstance(feats, list): feats = [feats]
            if any(Interval(f[0], f[1]).overlaps(other) for f in feats):
                ovls.append(ftype[:-1] if ftype[-1] == 's' else ftype)
        if 'cds' in ovls:
            ovls = [ft for ft in ovls if ft != 'exon']
        if self.cdsStart == self.cdsEnd:
            ovls = ['nc_' + ft for ft in ovls]
        return ovls