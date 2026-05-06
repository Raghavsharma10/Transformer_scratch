def parent_chain(self):
        """
        Return the list of parents starting from this node. The chain ends
        at the first node with no parents.
        """
        chain = [self]
        while True:
            try:
                parent = chain[-1].parent
            except Exception:
                break
            if parent is None:
                break
            chain.append(parent)
        return chain