def update(self, other):
        """Extend the current cluster with data from another cluster"""
        Cluster.update(self, other)
        self.rules.extend(other.rules)