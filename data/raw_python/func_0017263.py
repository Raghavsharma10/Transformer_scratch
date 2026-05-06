def hash_trees(self):
        "hash ladderized tree topologies"       
        observed = {}
        for idx, tree in enumerate(self.treelist):
            nwk = tree.write(tree_format=9)
            hashed = md5(nwk.encode("utf-8")).hexdigest()
            if hashed not in observed:
                observed[hashed] = idx
                self.treedict[idx] = 1
            else:
                idx = observed[hashed]
                self.treedict[idx] += 1