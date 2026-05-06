def runner(self):
        """
        Run the necessary methods in the correct order
        """
        printtime('Starting mashsippr analysis pipeline', self.starttime)
        if not self.pipeline:
            # Create the objects to be used in the analyses
            objects = Objectprep(self)
            objects.objectprep()
            self.runmetadata = objects.samples
        # Run the analyses
        Mash(self, self.analysistype)