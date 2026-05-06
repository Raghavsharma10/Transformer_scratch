def reporter(self):
        """
        Runs the necessary methods to parse raw read outputs
        """
        logging.info('Preparing reports')
        # Populate self.plusdict in order to reuse parsing code from an assembly-based method
        for sample in self.runmetadata.samples:
            self.plusdict[sample.name] = dict()
            self.matchdict[sample.name] = dict()
            if sample.general.bestassemblyfile != 'NA':
                for gene in sample[self.analysistype].allelenames:
                    self.plusdict[sample.name][gene] = dict()
                    for allele, percentidentity in sample[self.analysistype].results.items():

                        if gene in allele:
                            # Split the allele number from the gene name using the appropriate delimiter
                            if '_' in allele:
                                splitter = '_'
                            elif '-' in allele:
                                splitter = '-'
                            else:
                                splitter = ''
                            self.matchdict[sample.name].update({gene: allele.split(splitter)[-1]})
                            # Create the plusdict dictionary as in the assembly-based (r)MLST method. Allows all the
                            # parsing and sequence typing code to be reused.
                            try:
                                self.plusdict[sample.name][gene][allele.split(splitter)[-1]][percentidentity] \
                                    = sample[self.analysistype].avgdepth[allele]
                            except KeyError:
                                self.plusdict[sample.name][gene][allele.split(splitter)[-1]] = dict()
                                self.plusdict[sample.name][gene][allele.split(splitter)[-1]][percentidentity] \
                                    = sample[self.analysistype].avgdepth[allele]
                    if gene not in self.matchdict[sample.name]:
                        self.matchdict[sample.name].update({gene: 'N'})
        self.profiler()
        self.sequencetyper()
        self.mlstreporter()