def methodreporter(self):
        """
        Create final reports collating results from all the individual iterations through the method pipeline
        """
        # Ensure that the analyses are set to complete
        self.analysescomplete = True
        # Reset the report path to original value
        self.reportpath = os.path.join(self.path, 'reports')
        # Clear the runmetadata - it will be populated with all the metadata from completemetadata
        self.runmetadata = MetadataObject()
        self.runmetadata.samples = list()
        # As the samples were entered into self.completemetadata depending on when they passed the quality threshold,
        # this list is not ordered numerically/alphabetically like the original runmetadata. Reset the order.
        for strain in self.samples:
            for sample in self.completemetadata:
                if sample.name == strain:
                    # Append the sample to the ordered list of objects
                    self.runmetadata.samples.append(sample)
        # Create the reports
        self.reporter()
        self.genusspecific()
        self.sixteensreporter()
        self.gdcsreporter()
        self.confindr_reporter()