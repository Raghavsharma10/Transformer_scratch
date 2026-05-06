def similarity(self, other):
        """Calculate similarity based on best matching permutation of items."""
        # Select the longer list as the basis for comparison
        if len(self.items) > len(other.items):
            first, second = self, other
        else:
            first, second = other, self
        items = list(first.items)  # backup items list
        length = len(items)
        sim = self.Similarity(0.0 if length else 1.0)

        # Calculate the similarity for each permutation of items
        cname = self.__class__.__name__

        for num, perm in enumerate(permutations(items, length), start=1):
            first.items = perm
            aname = 'items-p{}'.format(num)
            self.log(first, second, '%', cname=cname, aname=aname)
            permutation_sim = super(Group, first).similarity(second)
            self.log(first, second, '%', cname=cname, aname=aname,
                     result=permutation_sim)

            sim = max(sim, permutation_sim)
            logging.debug("highest similarity: %s", sim)

        first.items = items  # restore original items list

        return sim