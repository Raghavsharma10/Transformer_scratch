def get_treenodes(self):
        "test format of intree nex/nwk, extra features"

        if not self.multitree:
            # get TreeNodes from Newick
            extractor = Newick2TreeNode(self.data[0].strip(), fmt=self.fmt)
        
            # extract one tree
            self.treenodes.append(extractor.newick_from_string())

        else:
            for tre in self.data:
                # get TreeNodes from Newick
                extractor = Newick2TreeNode(tre.strip(), fmt=self.fmt)
        
                # extract one tree
                self.treenodes.append(extractor.newick_from_string())