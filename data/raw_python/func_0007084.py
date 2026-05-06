def main(self):
        """
        Run the necessary methods in the correct order
        """
        self.target_validate()
        self.gene_names()
        Sippr(inputobject=self,
              k=self.kmer_size,
              allow_soft_clips=self.allow_soft_clips)
        self.report()