def _make_index(self, fasta, index):
        """ Index a single, one-sequence fasta-file"""
        out = open(index, "wb")
        f = open(fasta)
        # Skip first line of fasta-file
        line = f.readline()
        offset = f.tell()
        line = f.readline()
        while line:
            out.write(pack(self.pack_char, offset))
            offset = f.tell()
            line = f.readline()
        f.close()
        out.close()