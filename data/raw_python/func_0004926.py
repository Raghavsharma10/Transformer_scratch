def loadmask(self, filename: str) -> np.ndarray:
        """Load a mask file."""
        mask = scipy.io.loadmat(self.find_file(filename, what='mask'))
        maskkey = [k for k in mask.keys() if not (k.startswith('_') or k.endswith('_'))][0]
        return mask[maskkey].astype(np.bool)