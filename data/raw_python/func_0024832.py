def sampleset(self):
        """Return ``x`` array that samples the feature."""
        x1, x4 = self.bounding_box
        dw = self.width * 0.5
        x2 = self.x_0 - dw
        x3 = self.x_0 + dw

        if self._n_models == 1:
            w = [x1, x2, x3, x4]
        else:
            w = list(zip(x1, x2, x3, x4))

        return np.asarray(w)