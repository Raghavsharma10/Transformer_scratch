def transforms(self):
        """Return an array of arrays of column transforms.

        #The return value is an list of list, with each list being a segment of column transformations, and
        #each segment having one entry per column.

        """

        tr = []
        for c in self.columns:
            tr.append(c.expanded_transform)

        return six.moves.zip_longest(*tr)