def blastparse(self):
        """
        Parse the blast results, and store necessary data in dictionaries in sample object
        """
        logging.info('Parsing BLAST results')
        # Load the NCBI 16S reference database as a dictionary
        for sample in self.runmetadata.samples:
            if sample.general.bestassemblyfile != 'NA':
                # Load the NCBI 16S reference database as a dictionary
                dbrecords = SeqIO.to_dict(SeqIO.parse(sample[self.analysistype].baitfile, 'fasta'))
                # Allow for no BLAST results
                if os.path.isfile(sample[self.analysistype].blastreport):
                    # Initialise a dictionary to store the number of times a genus is the best hit
                    sample[self.analysistype].frequency = dict()
                    # Open the sequence profile file as a dictionary
                    blastdict = DictReader(open(sample[self.analysistype].blastreport),
                                           fieldnames=self.fieldnames, dialect='excel-tab')
                    recorddict = dict()
                    for record in blastdict:
                        # Create the subject id. It will look like this: gi|1018196593|ref|NR_136472.1|
                        subject = record['subject_id']
                        # Extract the genus name. Use the subject id as a key in the dictionary of the reference db.
                        # It will return the full record e.g. gi|1018196593|ref|NR_136472.1| Escherichia marmotae
                        # strain HT073016 16S ribosomal RNA, partial sequence
                        # This full description can be manipulated to extract the genus e.g. Escherichia
                        genus = dbrecords[subject].description.split('|')[-1].split()[0]
                        # Increment the number of times this genus was found, or initialise the dictionary with this
                        # genus the first time it is seen
                        try:
                            sample[self.analysistype].frequency[genus] += 1
                        except KeyError:
                            sample[self.analysistype].frequency[genus] = 1
                        try:
                            recorddict[dbrecords[subject].description] += 1
                        except KeyError:
                            recorddict[dbrecords[subject].description] = 1
                    # Sort the dictionary based on the number of times a genus is seen
                    sample[self.analysistype].sortedgenera = sorted(sample[self.analysistype].frequency.items(),
                                                                    key=operator.itemgetter(1), reverse=True)
                    try:
                        # Extract the top result, and set it as the genus of the sample
                        sample[self.analysistype].genus = sample[self.analysistype].sortedgenera[0][0]
                        # Previous code relies on having the closest refseq genus, so set this as above
                        # sample.general.closestrefseqgenus = sample[self.analysistype].genus
                    except IndexError:
                        # Populate attributes with 'NA'
                        sample[self.analysistype].sortedgenera = 'NA'
                        sample[self.analysistype].genus = 'NA'
                        # sample.general.closestrefseqgenus = 'NA'
                else:
                    # Populate attributes with 'NA'
                    sample[self.analysistype].sortedgenera = 'NA'
                    sample[self.analysistype].genus = 'NA'
                    # sample.general.closestrefseqgenus = 'NA'
            else:
                # Populate attributes with 'NA'
                sample[self.analysistype].sortedgenera = 'NA'
                sample[self.analysistype].genus = 'NA'