def out_of_bag_samples(self):
        """
        Returns the out-of-bag samples list, inside a wrapper to keep track
        of modifications.
        """
        #TODO:replace with more a generic pass-through wrapper?
        class O(object):
            def __init__(self, tree):
                self.tree = tree
            def __len__(self):
                return len(self.tree._out_of_bag_samples)
            def append(self, v):
                self.tree._out_of_bag_mae_clean = False
                return self.tree._out_of_bag_samples.append(v)
            def pop(self, v):
                self.tree._out_of_bag_mae_clean = False
                return self.tree._out_of_bag_samples.pop(v)
            def __iter__(self):
                for _ in self.tree._out_of_bag_samples:
                    yield _
        return O(self)