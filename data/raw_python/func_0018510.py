def promoter(self, up=2000, down=0):
        """
        Return a start, end tuple of positions for the promoter region of this
        gene

        Parameters
        ----------

        up : int
           this distance upstream that is considered the promoter

        down : int
           the strand is used to add this many downstream bases into the gene.
        """
        if not self.is_gene_pred: return None
        return self.tss(up=up, down=down)