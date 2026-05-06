def run_mash(self):
        """
        Run MASH to determine the closest refseq genomes
        """
        self.pipeline = True
        mash.Mash(inputobject=self,
                  analysistype='mash')