def save_feature(self, cat, img, feature, data):
        """Saves a new feature."""
        filename = self.path(cat, img, feature)
        mkdir(filename)
        savemat(filename, {'output':data})