def runner(self):
        """
        Run the necessary methods in the correct order
        """
        printtime('Starting {} analysis pipeline'.format(self.analysistype), self.starttime)
        if not self.pipeline:
            # If the metadata has been passed from the method script, self.pipeline must still be false in order to
            # get Sippr() to function correctly, but the metadata shouldn't be recreated
            try:
                _ = vars(self.runmetadata)['samples']
            except KeyError:
                # Create the objects to be used in the analyses
                objects = Objectprep(self)
                objects.objectprep()
                self.runmetadata = objects.samples
            # Run the analyses
            Sippr(self, self.cutoff)
            #
            self.attributer()
            # Create the reports
            self.reporter()
            # Print the metadata
            printer = MetadataPrinter(self)
            printer.printmetadata()