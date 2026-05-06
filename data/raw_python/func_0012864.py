def addinnode(self, otherplus, node, objectname):
        """add an item to the node.
        example: add a new zone to the element 'ZONE' """
        # do a test for unique object here
        newelement = otherplus.dt[node.upper()]