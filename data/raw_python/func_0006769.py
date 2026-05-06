def genusspecific(self, analysistype='genesippr'):
        """
        Creates simplified genus-specific reports. Instead of the % ID and the fold coverage, a simple +/- scheme is
        used for presence/absence
        :param analysistype: The variable to use when accessing attributes in the metadata object
        """
        # Dictionary to store all the output strings
        results = dict()
        for genus, genelist in self.genedict.items():
            # Initialise the dictionary with the appropriate genus
            results[genus] = str()
            for sample in self.runmetadata.samples:
                try:
                    # Find the samples that match the current genus - note that samples with multiple hits will be
                    # represented in multiple outputs
                    if genus in sample[analysistype].targetgenera:
                        # Populate the results string with the sample name
                        results[genus] += '{},'.format(sample.name)
                        # Iterate through all the genes associated with this genus. If the gene is in the current
                        # sample, add a + to the string, otherwise, add a -
                        for gene in genelist:
                            if gene.lower() in [target[0].lower().split('_')[0] for target in
                                                sample[analysistype].results.items()]:
                                results[genus] += '+,'
                            else:
                                results[genus] += '-,'
                        results[genus] += '\n'
                # If the sample is missing the targetgenera attribute, then it is ignored for these reports
                except AttributeError:
                    pass
        # Create and populate the genus-specific reports
        for genus, resultstring in results.items():
            # Only create the report if there are results for the current genus
            if resultstring:
                with open(os.path.join(self.reportpath, '{}_genesippr.csv'.format(genus)), 'w') as genusreport:
                    # Write the header to the report - Strain plus add the genes associated with the genus
                    genusreport.write('Strain,{}\n'.format(','.join(self.genedict[genus])))
                    # Write the results to the report
                    genusreport.write(resultstring)