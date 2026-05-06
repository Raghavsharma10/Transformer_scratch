def assembly_length(self):
        """
        Use SeqIO.parse to extract the total number of bases in each assembly file
        """
        for sample in self.metadata:
            # Only determine the assembly length if is has not been previously calculated
            if not GenObject.isattr(sample, 'assembly_length'):
                # Create the assembly_length attribute, and set it to 0
                sample.assembly_length = 0
                for record in SeqIO.parse(sample.bestassemblyfile, 'fasta'):
                    # Update the assembly_length attribute with the length of the current contig
                    sample.assembly_length += len(record.seq)
                # Write the updated object to file
                self.write_json(sample)