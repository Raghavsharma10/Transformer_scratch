def similarity(self, other):
        """Compare two objects for similarity.

        @param self: first object to compare
        @param other: second object to compare

        @return: L{Similarity} result of comparison

        """
        sim = self.Similarity()
        total = 0.0

        # Calculate similarity ratio for each attribute
        cname = self.__class__.__name__
        for aname, weight in self.attributes.items():

            attr1 = getattr(self, aname, None)
            attr2 = getattr(other, aname, None)
            self.log(attr1, attr2, '%', cname=cname, aname=aname)

            # Similarity is ignored if None on both objects
            if attr1 is None and attr2 is None:
                self.log(attr1, attr2, '%', cname=cname, aname=aname,
                         result="attributes are both None")
                continue

            # Similarity is 0 if either attribute is non-Comparable
            if not all((isinstance(attr1, Comparable),
                        isinstance(attr2, Comparable))):
                self.log(attr1, attr2, '%', cname=cname, aname=aname,
                         result="attributes not Comparable")
                total += weight
                continue

            # Calculate similarity between the attributes
            attr_sim = (attr1 % attr2)
            self.log(attr1, attr2, '%', cname=cname, aname=aname,
                     result=attr_sim)

            # Add the similarity to the total
            sim += attr_sim * weight
            total += weight

        # Scale the similarity so the total is 1.0
        if total:
            sim *= (1.0 / total)

        return sim