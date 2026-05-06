def tss(self, up=0, down=0):
        """
        Return a start, end tuple of positions around the transcription-start
        site

        Parameters
        ----------

        up : int
           if greature than 0, the strand is used to add this many upstream
           bases in the appropriate direction

        down : int
           if greature than 0, the strand is used to add this many downstream
           bases into the gene.
        """
        if not self.is_gene_pred: return None
        tss = self.txEnd if self.strand == '-' else self.txStart
        start, end = tss, tss
        if self.strand == '+':
            start -= up
            end += down
        else:
            start += up
            end -= down
            start, end = end, start
        return max(0, start), max(end, start, 0)