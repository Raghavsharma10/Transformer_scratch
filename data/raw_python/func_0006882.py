def runner(self):
        """
        Run the necessary methods in the correct order
        """
        printtime('Starting {} analysis pipeline'.format(self.analysistype), self.starttime)
        # Create the objects to be used in the analyses
        objects = Objectprep(self)
        objects.objectprep()
        self.runmetadata = objects.samples
        # Run the analyses
        sippr = Sippr(self, self.cutoff)
        sippr.clear()
        # Print the metadata
        printer = MetadataPrinter(self)
        printer.printmetadata()