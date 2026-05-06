def _bg_combine(self, bgs):
        """Combine several background amplitude images"""
        out = np.ones(self.h5["raw"].shape, dtype=float)
        # bg is an h5py.DataSet
        for bg in bgs:
            out *= bg[:]
        return out