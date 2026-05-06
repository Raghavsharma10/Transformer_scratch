def to_motevo(self):
        """Return motif formatted in MotEvo (TRANSFAC-like) format
        
        Returns
        -------
        m : str
            String of motif in MotEvo format.
        """
        m = "//\n"
        m += "NA {}\n".format(self.id)
        m += "P0\tA\tC\tG\tT\n"
        for i, row in enumerate(self.pfm):
            m += "{}\t{}\n".format(i, "\t".join([str(int(x)) for x in row]))
        m += "//"
        return m