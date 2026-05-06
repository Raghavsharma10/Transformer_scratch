def f(self, y, t):
        """Deterministic term f of the complete network system
        dy = f(y, t)dt + G(y, t).dot(dW)

        (or for an ODE network system without noise, dy/dt = f(y, t))

        Args:
          y (array of shape (d,)): where d is the dimension of the overall
            state space of the complete network system. 

        Returns: 
          f (array of shape (d,)):  Defines the deterministic term of the
            complete network system
        """
        coupling = self.coupling_function[0]
        res = np.empty_like(self.y0)
        for j, m in enumerate(self.submodels):
            slicej = slice(self._si[j], self._si[j+1])
            target_y = y[slicej] # target node state
            res[slicej] = m.f(target_y, t) # deterministic part of submodel j
            # get indices of all source nodes that provide input to node j:
            sources = np.nonzero(self.network[:,j])[0]
            for i in sources:
                weight = self.network[i, j]
                source_y = y[slice(self._si[i], self._si[i+1])] # source state
                res[slicej] += coupling(source_y, target_y, weight)
        return res