def _read_seq_from_fasta(self, fasta, offset, nr_lines):
        """ retrieve a number of lines from a fasta file-object, starting at offset"""
        fasta.seek(offset)
        lines = [fasta.readline().strip() for _ in range(nr_lines)]
        return "".join(lines)