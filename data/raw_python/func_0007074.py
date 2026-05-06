def runner(self):
        """
        Run the necessary methods in the correct order
        """
        logging.info('Starting {} analysis pipeline'.format(self.analysistype))
        # Run the analyses
        Sippr(self, self.cutoff)
        self.serotype_escherichia()
        self.serotype_salmonella()
        # Create the reports
        self.reporter()
        # Print the metadata
        metadataprinter.MetadataPrinter(self)