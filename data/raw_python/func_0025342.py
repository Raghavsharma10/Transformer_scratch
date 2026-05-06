def set_classifier_mask(self, v, base_mask=True):
        """Computes the mask used to create the training and validation set"""
        base = self._base
        v = tonparray(v)
        a = np.unique(v)
        if a[0] != -1 or a[1] != 1:
            raise RuntimeError("The labels must be -1 and 1 (%s)" % a)
        mask = np.zeros_like(v)
        cnt = min([(v == x).sum() for x in a]) * base._tr_fraction
        cnt = int(round(cnt))
        for i in a:
            index = np.where((v == i) & base_mask)[0]
            np.random.shuffle(index)
            mask[index[:cnt]] = True
        base._mask = SparseArray.fromlist(mask)
        return SparseArray.fromlist(v)