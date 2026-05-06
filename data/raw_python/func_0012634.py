def fixations(self):
        ''' Filter the fixmat such that it only contains fixations on images
        in categories that are also in the categories object'''
        if not self._fixations:
            raise RuntimeError('This Images object does not have'
                +' an associated fixmat')
        if len(list(self._categories.keys())) == 0:
            return None
        else:
            idx = np.zeros(self._fixations.x.shape, dtype='bool')
            for (cat, _) in list(self._categories.items()):
                idx = idx | ((self._fixations.category == cat))
            return self._fixations[idx]