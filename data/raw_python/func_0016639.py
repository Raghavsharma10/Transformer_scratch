def train(self, record):
        """
        Updates the trees with the given training record.
        """
        self._fell_trees()
        self._grow_trees()
        for tree in self.trees:
            if random.random() < self.sample_ratio:
                tree.train(record)
            else:
                tree.out_of_bag_samples.append(record)
                while len(tree.out_of_bag_samples) > self.max_out_of_bag_samples:
                    tree.out_of_bag_samples.pop(0)