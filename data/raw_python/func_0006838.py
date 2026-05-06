def runner(self):
        """
        Run the necessary methods in the correct order
        """
        logging.info('Starting {} analysis pipeline'.format(self.analysistype))
        if not self.pipeline:
            # If the metadata has been passed from the method script, self.pipeline must still be false in order to
            # get Sippr() to function correctly, but the metadata shouldn't be recreated
            try:
                _ = vars(self.runmetadata)['samples']
            except AttributeError:
                # Create the objects to be used in the analyses
                objects = Objectprep(self)
                objects.objectprep()
                self.runmetadata = objects.samples

        else:
            for sample in self.runmetadata.samples:
                setattr(sample, self.analysistype, GenObject())
                sample.run.outputdirectory = sample.general.outputdirectory
        self.threads = int(self.cpus / len(self.runmetadata.samples)) \
            if self.cpus / len(self.runmetadata.samples) > 1 \
            else 1
        # Use a custom sippr method to use the full reference database as bait, and run mirabait against the FASTQ
        # reads - do not perform reference mapping yet
        SixteenSBait(self, self.cutoff)
        # Subsample 1000 reads from the FASTQ files
        self.subsample()
        # Convert the subsampled FASTQ files to FASTA format
        self.fasta()
        # Create BLAST databases if required
        self.makeblastdb()
        # Run BLAST analyses of the subsampled FASTA files against the NCBI 16S reference database
        self.blast()
        # Parse the BLAST results
        self.blastparse()
        # Feed the BLAST results into a modified sippr method to perform reference mapping using the calculated
        # genus of the sample as the mapping file
        SixteenSSipper(self, self.cutoff)
        # Create reports
        self.reporter()