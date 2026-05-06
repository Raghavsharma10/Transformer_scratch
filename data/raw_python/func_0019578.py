def writefasta(self, fname):
        """ Write sequences to FASTA formatted file"""
        f = open(fname, "w")
        fa_str = "\n".join([">%s\n%s" % (id, self._format_seq(seq)) for id, seq in self.items()])
        f.write(fa_str)
        f.close()