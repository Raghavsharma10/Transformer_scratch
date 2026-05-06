def build(self):
        """Builds the tree from the leaves that have been added.

        This function populates the tree from the leaves down non-recursively
        """
        self.order = MerkleTree.get_order(len(self.leaves))
        n = 2 ** self.order
        self.nodes = [b''] * 2 * n

        # populate lowest nodes with leaf hashes
        for j in range(0, n):
            if (j < len(self.leaves)):
                self.nodes[j + n - 1] = self.leaves[j].get_hash()
            else:
                break

        # now populate the entire tree
        for i in range(1, self.order + 1):
            p = 2 ** (self.order - i)
            for j in range(0, p):
                k = p + j - 1
                h = hashlib.sha256()
                l = self.nodes[MerkleTree.get_left_child(k)]
                if (len(l) > 0):
                    h.update(l)
                r = self.nodes[MerkleTree.get_right_child(k)]
                if (len(r) > 0):
                    h.update(r)
                self.nodes[k] = h.digest()