def get_sequences(self, chr, coords):
        """ Retrieve multiple sequences from same chr (RC not possible yet)"""    
        # Check if we have an index_dir
        if not self.index_dir:
            print("Index dir is not defined!")
            sys.exit()

        # retrieve all information for this specific sequence
        fasta_file = self.fasta_file[chr]
        index_file = self.index_file[chr]
        line_size = self.line_size[chr]
        total_size = self.size[chr]
        index = open(index_file, "rb")
        fasta = open(fasta_file)
        
        seqs = []
        for coordset in coords:
            seq = ""
            for (start,end) in coordset: 
                if start > total_size:
                    raise ValueError("%s: %s, invalid start, greater than sequence length!" % (chr,start))
            
                if start < 0:
                    raise ValueError("Invalid start, < 0!")
                
                if end > total_size:
                    raise ValueError("Invalid end, greater than sequence length!")


                seq += self._read(index, fasta, start, end, line_size)
            seqs.append(seq)
        index.close()
        fasta.close()

        return seqs