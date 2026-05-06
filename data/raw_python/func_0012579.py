def path(self, category = None, image = None, feature = None):
        """
		Constructs the path to categories, images and features.

        This path function assumes that the following storage scheme is used on
        the hard disk to access categories, images and features:
            - categories: /impath/category
            - images:     /impath/category/category_image.png
            - features:   /ftrpath/category/feature/category_image.mat

        The path function is called to query the location of categories, images
        and features before they are loaded. Thus, if your features are organized
        in a different way, you can simply replace this method such that it returns
        appropriate paths' and the LoadFromDisk loader will use your naming
        scheme.
		"""
        filename = None
        if not category is None:
            filename = join(self.impath, str(category))
        if not image is None:
            assert not category is None, "The category has to be given if the image is given"
            filename = join(filename,
                '%s_%s.png' % (str(category), str(image)))
        if not feature is None:
            assert category != None and image != None, "If a feature name is given the category and image also have to be given."
            filename = join(self.ftrpath, str(category), feature,
                '%s_%s.mat' % (str(category), str(image)))
        return filename