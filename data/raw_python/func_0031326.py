def transform(self, images):
        """
        Apply the transformation to an Images object.

        Will apply the underlying dictionary of transformations to
        the images or volumes of the Images object. The dictionary acts as a lookup
        table specifying which transformation should be applied to which record of the
        Images object based on the key. Because transformations are small,
        we broadcast the transformations rather than using a join.

        Parameters
        ----------
        images : array-like or thunder images
            The sequence of images / volumes to register.
        """
        images = check_images(images)

        def apply(item):
            (k, v) = item
            return self.transformations[k].apply(v)

        return images.map(apply, value_shape=images.value_shape, dtype=images.dtype, with_keys=True)