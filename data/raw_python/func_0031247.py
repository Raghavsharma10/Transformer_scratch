def fit(self, images, reference=None):
        """
        Estimate registration model using cross-correlation.

        Use cross correlation to compute displacements between 
        images or volumes and reference. Displacements will be 
        2D for images and 3D for volumes.

        Parameters
        ----------
        images : array-like or thunder images
            The sequence of images / volumes to register.

        reference : array-like
            A reference image to align to.
        """
        images = check_images(images)
        reference = check_reference(images, reference)

        def func(item):
            key, image = item
            return asarray([key, self._get(image, reference)])

        transformations = images.map(func, with_keys=True).toarray()
        if images.shape[0] == 1: 
            transformations = [transformations]

        algorithm = self.__class__.__name__
        return RegistrationModel(dict(transformations), algorithm=algorithm)