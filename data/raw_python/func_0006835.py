def main(self):
        """
        Run the required methods in the appropriate order
        """
        self.targets()
        self.bait(k=49)
        self.reversebait(maskmiddle='t', k=19)
        self.subsample_reads()