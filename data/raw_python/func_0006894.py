def attributer(self):
        """
        Parses the 16S target files to link accession numbers stored in the .fai and metadata files to the genera stored
        in the target file
        """
        from Bio import SeqIO
        import operator
        for sample in self.runmetadata.samples:
            # Load the records from the target file into a dictionary
            record_dict = SeqIO.to_dict(SeqIO.parse(sample[self.analysistype].baitfile, "fasta"))
            sample[self.analysistype].classification = set()
            sample[self.analysistype].genera = dict()
            # Add all the genera with hits into the set of genera
            for result in sample[self.analysistype].results:
                genus, species = record_dict[result].description.split('|')[-1].split()[:2]
                sample[self.analysistype].classification.add(genus)
                sample[self.analysistype].genera[result] = genus
            # Convert the set to a list for easier JSON serialisation
            sample[self.analysistype].classification = list(sample[self.analysistype].classification)
            # If there is a mixed sample, then further analyses will be complicated
            if len(sample[self.analysistype].classification) > 1:
                # print('multiple: ', sample.name, sample[self.analysistype].classification)
                sample.general.closestrefseqgenus = sample[self.analysistype].classification
                # sample.general.bestassemblyfile = 'NA'
                sample[self.analysistype].multiple = True
            else:
                sample[self.analysistype].multiple = False

                try:
                    # Recreate the results dictionary with the percent identity as a float rather than a string
                    sample[self.analysistype].intresults = \
                        {key: float(value) for key, value in sample[self.analysistype].results.items()}
                    # Set the best hit to be the top entry from the sorted results
                    sample[self.analysistype].besthit = sorted(sample[self.analysistype].intresults.items(),
                                                               key=operator.itemgetter(1), reverse=True)[0]
                    sample.general.closestrefseqgenus = sample[self.analysistype].classification[0]
                except IndexError:
                    sample.general.bestassemblyfile = 'NA'