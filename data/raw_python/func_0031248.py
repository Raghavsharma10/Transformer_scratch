def fit_and_transform(self, images, reference=None):
        """
        Estimate and apply registration model using cross-correlation.

        Use cross correlation to compute displacements between 
        images or volumes and reference, and apply the
        estimated model to the data. Displacements will be 
        2D for images and 3D for volumes.

        Parameters
        ----------
        images : array-like or thunder images
            The sequence of images / volumes to register.

        reference : array-like
            A reference image to align to.
        """
        images = check_images(images)
        check_reference(images, reference)

        def func(image):
            t = self._get(image, reference)
            return t.apply(image)

        return images.map(func)