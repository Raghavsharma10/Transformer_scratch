def G(self, y, t):
        """Noise coefficient matrix G of the complete network system
        dy = f(y, t)dt + G(y, t).dot(dW)

        (for an ODE network system without noise this function is not used)

        Args:
          y (array of shape (d,)): where d is the dimension of the overall
            state space of the complete network system. 

        Returns:
          G (array of shape (d, m)): where m is the number of independent
            Wiener processes driving the complete network system. The noise
            coefficient matrix G defines the stochastic term of the system.
        """
        if self._independent_noise:
            # then G matrix consists of submodel Gs diagonally concatenated:
            res = np.zeros((self.dimension, self.nnoises))
            offset = 0
            for j, m in enumerate(self.submodels):
                slicej = slice(self._si[j], self._si[j+1])
                ix = (slicej, slice(offset, offset + self._nsubnoises[j]))
                res[ix] = m.G(y[slicej], t) # submodel noise coefficient matrix
                offset += self._nsubnoises[j]
        else:
            # identical driving: G consists of submodel Gs stacked vertically
            res = np.empty((self.dimension, self.nnoises))
            for j, m in enumerate(self.submodels):
                slicej = slice(self._si[j], self._si[j+1])
                ix = (slicej, slice(None))
                res[ix] = m.G(y[slicej], t) # submodel noise coefficient matrix
        return res