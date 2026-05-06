def get_image(self, cat, img):
        """ Loads an image from disk. """
        filename = self.path(cat, img)
        data = []
        if filename.endswith('mat'):
            data = loadmat(filename)['output']
        else:
            data = imread(filename)
        if self.size is not None:
            return imresize(data, self.size)
        else:
            return data