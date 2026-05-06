def add_feature(self, pr_name, pr_value):
        """ Add or update a node's feature. """
        setattr(self, pr_name, pr_value)
        self.features.add(pr_name)