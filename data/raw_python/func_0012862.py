def replacenode(self, othereplus, node):
        """replace the node here with the node from othereplus"""
        node = node.upper()
        self.dt[node.upper()] = othereplus.dt[node.upper()]