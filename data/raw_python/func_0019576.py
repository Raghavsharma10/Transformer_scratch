def hardmask(self):
        """ Mask all lowercase nucleotides with N's """
        p = re.compile("a|c|g|t|n")
        for seq_id in self.fasta_dict.keys():
            self.fasta_dict[seq_id] = p.sub("N", self.fasta_dict[seq_id])
        return self