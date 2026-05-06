def get_sequence(self, chrom, start, end, strand=None):
        """ Retrieve a sequence """    
        # Check if we have an index_dir
        if not self.index_dir:
            print("Index dir is not defined!")
            sys.exit()

        # retrieve all information for this specific sequence
        fasta_file = self.fasta_file[chrom]
        index_file = self.index_file[chrom]
        line_size = self.line_size[chrom]
        total_size = self.size[chrom]

        #print fasta_file, index_file, line_size, total_size
        if start > total_size:
            raise ValueError(
                    "Invalid start {0}, greater than sequence length {1} of {2}!".format(start, total_size, chrom))
        
        if start < 0:
            raise ValueError("Invalid start, < 0!")
        
        if end > total_size:
            raise ValueError(
                    "Invalid end {0}, greater than sequence length {1} of {2}!".format(end, total_size, chrom))


        index = open(index_file, "rb")
        fasta = open(fasta_file)
        seq = self._read(index, fasta, start, end, line_size)
        index.close()
        fasta.close()

        if strand and strand == "-":
            seq = rc(seq)
        return seq