def genus_specific(self):
        """
        For genus-specific targets, MLST and serotyping, determine if the closest refseq genus is known - i.e. if 16S
        analyses have been performed. Perform the analyses if required
        """
        # Initialise a variable to store whether the necessary analyses have already been performed
        closestrefseqgenus = False
        for sample in self.runmetadata.samples:
            if sample.general.bestassemblyfile != 'NA':
                try:
                    closestrefseqgenus = sample.general.closestrefseqgenus
                except AttributeError:
                    pass
        # Perform the 16S analyses as required
        if not closestrefseqgenus:
            logging.info('Must perform MASH analyses to determine genera of samples')
            self.pipeline = True
            # Run the analyses
            mash.Mash(self, 'mash')