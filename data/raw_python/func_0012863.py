def add2node(self, othereplus, node):
        """add the node here with the node from othereplus
        this will potentially have duplicates"""
        node = node.upper()
        self.dt[node.upper()] = self.dt[node.upper()] + \
            othereplus.dt[node.upper()]