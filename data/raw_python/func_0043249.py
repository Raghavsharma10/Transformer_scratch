def _gradient(self, diff, d, coords):
        """Compute the gradient.

        Args:
            diff (`array-like`): [`m`, `m`] matrix. `D` - `d`
            d (`array-like`): [`m`, `m`] matrix.
            coords (`array-like`): [`m`, `n`] matrix.

        Returns:
            `np.array`: Gradient, shape [`m`, `n`].
        """
        denom = np.copy(d)
        denom[denom == 0] = 1e-5

        with np.errstate(divide='ignore', invalid='ignore'):
            K = -2 * diff / denom

        K[np.isnan(K)] = 0

        g = np.empty_like(coords)
        for n in range(self.n):
            for i in range(self.m):
                # Vectorised version of (~70 times faster)
                # for j in range(self.m):
                #     delta_g = ((coords[i, n] - coords[j, n]) * K[i, j]).sum()
                #     g[i, n] += delta_g
                g[i, n] = ((coords[i, n] - coords[:, n]) * K[i, :]).sum()

        return g