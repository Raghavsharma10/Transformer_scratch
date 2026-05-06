def fixations(self):
        """
        Returns all fixations that are on this image.
        A precondition for this to work is that a fixmat 
        is associated with this Image object.
        """
        if not self._fixations:
            raise RuntimeError('This Images object does not have'
                +' an associated fixmat')
        return self._fixations[(self._fixations.category == self.category) &
                               (self._fixations.filenumber == self.image)]