def find_clades(self):
        "Count clade occurrences."
        # index names from the first tree
        ndict = {j: i for i, j in enumerate(self.names)}
        namedict = {i: j for i, j in enumerate(self.names)}

        # store counts
        clade_counts = {}
        for tidx, ncopies in self.treedict.items():
            
            # testing on unrooted trees is easiest but for some reason slow
            ttree = self.treelist[tidx].unroot()

            # traverse over tree
            for node in ttree.treenode.traverse('preorder'):
                bits = np.zeros(len(ttree), dtype=np.bool_)
                for child in node.iter_leaf_names():
                    bits[ndict[child]] = True

                # get bit string and its reverse
                bitstring = bits.tobytes()
                revstring = np.invert(bits).tobytes()

                # add to clades first time, then check for inverse next hits
                if bitstring in clade_counts:
                    clade_counts[bitstring] += ncopies
                else:
                    if revstring not in clade_counts:
                        clade_counts[bitstring] = ncopies
                    else:
                        clade_counts[revstring] += ncopies

        # convert to freq
        for key, val in clade_counts.items():
            clade_counts[key] = val / float(len(self.treelist))

        ## return in sorted order
        self.namedict = namedict
        self.clade_counts = sorted(
            clade_counts.items(),
            key=lambda x: x[1],
            reverse=True)