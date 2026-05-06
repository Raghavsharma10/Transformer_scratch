def gene_features(self):
        """
        return a list of features for the gene features of this object.
        This would include exons, introns, utrs, etc.
        """
        nm, strand = self.gene_name, self.strand
        feats = [(self.chrom, self.start, self.end, nm, strand, 'gene')]
        for feat in ('introns', 'exons', 'utr5', 'utr3', 'cdss'):
            fname = feat[:-1] if feat[-1] == 's' else feat
            res = getattr(self, feat)
            if res is None or all(r is None for r in res): continue
            if not isinstance(res, list): res = [res]
            feats.extend((self.chrom, s, e, nm, strand, fname) for s, e in res)

        tss = self.tss(down=1)
        if tss is not None:
            feats.append((self.chrom, tss[0], tss[1], nm, strand, 'tss'))
            prom = self.promoter()
            feats.append((self.chrom, prom[0], prom[1], nm, strand, 'promoter'))

        return sorted(feats, key=itemgetter(1))