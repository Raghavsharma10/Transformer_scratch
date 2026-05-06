def set_regression_mask(self, v):
        """Computes the mask used to create the training and validation set"""
        base = self._base
        index = np.arange(v.size())
        np.random.shuffle(index)
        ones = np.ones(v.size())
        ones[index[int(base._tr_fraction * v.size()):]] = 0
        base._mask = SparseArray.fromlist(ones)