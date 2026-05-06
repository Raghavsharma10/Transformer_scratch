def toarray(self):
        """
        Return transformations as an array with shape (n,x1,x2,...)
        where n is the number of images, and remaining dimensions depend
        on the particular transformations
        """
        return asarray([value.toarray() for (key, value) in sorted(self.transformations.items())])