def runner(self):
        """
        Run the necessary methods in the correct order
        """
        logging.info('Starting {} analysis pipeline'.format(self.analysistype))
        # Initialise the GenObject
        for sample in self.runmetadata.samples:
            setattr(sample, self.analysistype, GenObject())
            try:
                sample[self.analysistype].pointfindergenus = self.pointfinder_org_dict[sample.general.referencegenus]
            except KeyError:
                sample[self.analysistype].pointfindergenus = 'ND'
        # Run the raw read mapping
        PointSipping(inputobject=self,
                     cutoff=self.cutoff)
        # Create FASTA files from the raw read matcves
        self.fasta()
        # Run PointFinder on the FASTA files
        self.run_pointfinder()
        # Create summary reports of the PointFinder outputs
        self.parse_pointfinder()