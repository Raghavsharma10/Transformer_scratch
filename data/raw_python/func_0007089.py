def runner(self):
        """
        Run the necessary methods in the correct order
        """
        logging.info('Starting {} analysis pipeline'.format(self.analysistype))
        if not self.pipeline:
            general = None
            for sample in self.runmetadata.samples:
                general = getattr(sample, 'general')
            if general is None:
                # Create the objects to be used in the analyses
                objects = Objectprep(self)
                objects.objectprep()
                self.runmetadata = objects.samples
        # Run the analyses
        Sippr(self, self.cutoff)
        # Create the reports
        reports = Reports(self)
        Reports.reporter(reports, analysistype=self.analysistype)
        # Print the metadata
        MetadataPrinter(self)