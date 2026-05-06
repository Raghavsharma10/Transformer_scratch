def get_feature(self, cat, img, feature):
        """
        Load a feature from disk.
        """
        filename = self.path(cat, img, feature)
        data = loadmat(filename)
        name = [k for k in list(data.keys()) if not k.startswith('__')]
        if self.size is not None:
            return imresize(data[name.pop()], self.size)
        return data[name.pop()]