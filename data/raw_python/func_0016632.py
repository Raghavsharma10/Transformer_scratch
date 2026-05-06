def train(self, record):
        """
        Incrementally updates the tree with the given sample record.
        """
        assert self.data.class_attribute_name in record, \
            "The class attribute must be present in the record."
        record = record.copy()
        self.sample_count += 1
        self.tree.train(record)