def main(self):
        """
        Run the analyses using the inputted values for forward and reverse read length. However, if not all strains
        pass the quality thresholds, continue to periodically run the analyses on these incomplete strains until either
        all strains are complete, or the sequencing run is finished
        """
        logging.info('Starting {} analysis pipeline'.format(self.analysistype))
        self.createobjects()
        # Run the genesipping analyses
        self.methods()
        # Determine if the analyses are complete
        self.complete()
        self.additionalsipping()
        # Update the report object
        self.reports = Reports(self)
        # Once all the analyses are complete, create reports for each sample
        Reports.methodreporter(self.reports)
        # Print the metadata
        printer = MetadataPrinter(self)
        printer.printmetadata()