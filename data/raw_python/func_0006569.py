def complete(self):
        """
        Determine if the analyses of the strains are complete e.g. there are no missing GDCS genes, and the 
        sample.general.bestassemblyfile != 'NA'
        """
        # Boolean to store the completeness of the analyses
        allcomplete = True
        # Clear the list of samples that still require more sequence data
        self.incomplete = list()
        for sample in self.runmetadata.samples:
            if sample.general.bestassemblyfile != 'NA':
                try:
                    # If the sample has been tagged as incomplete, only add it to the complete metadata list if the
                    # pipeline is on its final iteration
                    if sample.general.incomplete:
                        if self.final:
                            self.completemetadata.append(sample)
                        else:
                            sample.general.complete = False
                            allcomplete = False
                            self.incomplete.append(sample.name)
                except AttributeError:
                    sample.general.complete = True
                    self.completemetadata.append(sample)
            else:
                if self.final:
                    self.completemetadata.append(sample)
                else:
                    sample.general.complete = False
                    allcomplete = False
                    self.incomplete.append(sample.name)
        # If all the samples are complete, set the global variable for run completeness to True
        if allcomplete:
            self.analysescomplete = True