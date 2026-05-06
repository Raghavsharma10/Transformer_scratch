def reporter(self):
        """
        Creates a report of the results
        """
        # Create the path in which the reports are stored
        make_path(self.reportpath)
        logging.info('Creating {} report'.format(self.analysistype))
        # Initialise the header and data strings
        header = 'Strain,Gene,PercentIdentity,Genus,FoldCoverage\n'
        data = ''
        with open(self.sixteens_report, 'w') as report:
            with open(os.path.join(self.reportpath, self.analysistype + '_sequences.fa'), 'w') as sequences:
                for sample in self.runmetadata.samples:
                    # Initialise
                    sample[self.analysistype].sixteens_match = 'NA'
                    sample[self.analysistype].species = 'NA'
                    try:
                        # Select the best hit of all the full-length 16S genes mapped - for 16S use the hit with the
                        # fewest number of SNPs rather than the highest percent identity
                        sample[self.analysistype].besthit = sorted(sample[self.analysistype].resultssnp.items(),
                                                                   key=operator.itemgetter(1))[0][0]
                        # Parse the baited FASTA file to pull out the the description of the hit
                        for record in SeqIO.parse(sample[self.analysistype].baitfile, 'fasta'):
                            # If the best hit e.g. gi|631251361|ref|NR_112558.1| is present in the current record,
                            # gi|631251361|ref|NR_112558.1| Escherichia coli strain JCM 1649 16S ribosomal RNA ...,
                            # extract the match and the species
                            if sample[self.analysistype].besthit in record.id:
                                # Set the best match and species from the records
                                sample[self.analysistype].sixteens_match = record.description.split(' 16S')[0]
                                sample[self.analysistype].species = \
                                    sample[self.analysistype].sixteens_match.split('|')[-1].split()[1]
                        # Add the sample name to the data string
                        data += sample.name + ','
                        # Find the record that matches the best hit, and extract the necessary values to be place in the
                        # data string
                        for name, identity in sample[self.analysistype].results.items():
                            if name == sample[self.analysistype].besthit:
                                data += '{},{},{},{}\n'.format(name, identity, sample[self.analysistype].genus,
                                                               sample[self.analysistype].avgdepth[name])
                                # Create a FASTA-formatted sequence output of the 16S sequence
                                record = SeqRecord(Seq(sample[self.analysistype].sequences[name],
                                                       IUPAC.unambiguous_dna),
                                                   id='{}_{}'.format(sample.name, '16S'),
                                                   description='')
                                SeqIO.write(record, sequences, 'fasta')
                    except (AttributeError, IndexError):
                        data += '{}\n'.format(sample.name)
            # Write the results to the report
            report.write(header)
            report.write(data)