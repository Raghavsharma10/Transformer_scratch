def fasta(self):
        """
        Create FASTA files of the PointFinder results to be fed into PointFinder
        """
        logging.info('Extracting FASTA sequences matching PointFinder database')
        for sample in self.runmetadata.samples:
            # Ensure that there are sequence data to extract from the GenObject
            if GenObject.isattr(sample[self.analysistype], 'sequences'):
                # Set the name of the FASTA file
                sample[self.analysistype].pointfinderfasta = \
                    os.path.join(sample[self.analysistype].outputdir,
                                 '{seqid}_pointfinder.fasta'.format(seqid=sample.name))
                # Create a list to store all the SeqRecords created
                sequences = list()
                with open(sample[self.analysistype].pointfinderfasta, 'w') as fasta:
                    for gene, sequence in sample[self.analysistype].sequences.items():
                        # Create a SeqRecord using a Seq() of the sequence - both SeqRecord and Seq are from BioPython
                        seq = SeqRecord(seq=Seq(sequence),
                                        id=gene,
                                        name=str(),
                                        description=str())
                        sequences.append(seq)
                    # Write all the SeqRecords to file
                    SeqIO.write(sequences, fasta, 'fasta')