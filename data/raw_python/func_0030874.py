def get_chain(self, index):
        """Assemble and return the chain leading from a given node to the merkle root of this tree.
        """
        chain = []
        this = self.leaves[index]
        chain.append((this.val, 'SELF'))
        while this.p:
            chain.append((this.sib.val, this.sib.side))
            this = this.p
        chain.append((this.val, 'ROOT'))
        return chain