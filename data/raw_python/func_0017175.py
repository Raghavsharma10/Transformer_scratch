def del_feature(self, pr_name):
        """ Permanently deletes a node's feature."""
        if hasattr(self, pr_name):
            delattr(self, pr_name)
            self.features.remove(pr_name)