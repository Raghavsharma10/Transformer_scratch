def fasta(self):
        """Generates sequence data for the protein in FASTA format."""
        max_line_length = 79
        fasta_str = '>{0}:{1}|PDBID|CHAIN|SEQUENCE\n'.format(
            self.parent.id.upper(), self.id)
        seq = self.sequence
        split_seq = [seq[i: i + max_line_length]
                     for i in range(0, len(seq), max_line_length)]
        for seq_part in split_seq:
            fasta_str += '{0}\n'.format(seq_part)
        return fasta_str