def sixteensreporter(self, analysistype='sixteens_full'):
        """
        Creates a report of the results
        :param analysistype: The variable to use when accessing attributes in the metadata object
        """
        # Create the path in which the reports are stored
        make_path(self.reportpath)
        # Initialise the header and data strings
        header = 'Strain,Gene,PercentIdentity,Genus,FoldCoverage\n'
        data = ''
        with open(os.path.join(self.reportpath, analysistype + '.csv'), 'w') as report:
            with open(os.path.join(self.reportpath, analysistype + '_sequences.fa'), 'w') as sequences:
                for sample in self.runmetadata.samples:
                    try:
                        # Select the best hit of all the full-length 16S genes mapped
                        sample[analysistype].besthit = sorted(sample[analysistype].results.items(),
                                                              key=operator.itemgetter(1), reverse=True)[0][0]
                        # Add the sample name to the data string
                        data += sample.name + ','
                        # Find the record that matches the best hit, and extract the necessary values to be place in the
                        # data string
                        for name, identity in sample[analysistype].results.items():
                            if name == sample[analysistype].besthit:
                                data += '{},{},{},{}\n'.format(name, identity, sample[analysistype].genus,
                                                               sample[analysistype].avgdepth[name])
                                # Create a FASTA-formatted sequence output of the 16S sequence
                                record = SeqRecord(Seq(sample[analysistype].sequences[name],
                                                       IUPAC.unambiguous_dna),
                                                   id='{}_{}'.format(sample.name, '16S'),
                                                   description='')
                                SeqIO.write(record, sequences, 'fasta')
                    except (KeyError, IndexError):
                        data += '{}\n'.format(sample.name)
            # Write the results to the report
            report.write(header)
            report.write(data)