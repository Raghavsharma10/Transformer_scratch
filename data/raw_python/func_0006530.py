def runner(self):
        """
        Run the necessary methods in the correct order
        """
        if os.path.isfile(self.report):
            self.report_parse()
        else:
            logging.info('Starting {} analysis pipeline'.format(self.analysistype))
            # Create the objects to be used in the analyses (if required)
            general = None
            for sample in self.runmetadata.samples:
                general = getattr(sample, 'general')
            if general is None:
                # Create the objects to be used in the analyses
                objects = Objectprep(self)
                objects.objectprep()
                self.runmetadata = objects.samples
            # Run the analyses
            MLSTmap(self, self.analysistype, self.cutoff)
            # Create the reports
            self.reporter()
            # Print the metadata to a .json file
            MetadataPrinter(self)