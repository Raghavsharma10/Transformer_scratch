def hessian_component(self, index1, index2):
        """Compute the hessian of the energy for one atom pair"""
        result = np.zeros((3, 3), float)
        if index1 == index2:
            for index3 in range(self.numc):
                if self.scaling[index1, index3] > 0:
                    d_1 = 1/self.distances[index1, index3]
                    for (se, ve), (sg, vg), (sh, vh) in zip(
                        self.yield_pair_energies(index1, index3),
                        self.yield_pair_gradients(index1, index3),
                        self.yield_pair_hessians(index1, index3)
                    ):
                        result += (
                            +sh*self.dirouters[index1, index3]*ve
                            +sg*(np.identity(3, float) - self.dirouters[index1, index3])*ve*d_1
                            +sg*np.outer(self.directions[index1, index3],  vg)
                            +sg*np.outer(vg, self.directions[index1, index3])
                            +se*vh
                        )*self.scaling[index1, index3]
        elif self.scaling[index1, index2] > 0:
            d_1 = 1/self.distances[index1, index2]
            for (se, ve), (sg, vg), (sh, vh) in zip(
                self.yield_pair_energies(index1, index2),
                self.yield_pair_gradients(index1, index2),
                self.yield_pair_hessians(index1, index2)
            ):
                result -= (
                    +sh*self.dirouters[index1, index2]*ve
                    +sg*(np.identity(3, float) - self.dirouters[index1, index2])*ve*d_1
                    +sg*np.outer(self.directions[index1, index2],  vg)
                    +sg*np.outer(vg, self.directions[index1, index2])
                    +se*vh
                )*self.scaling[index1, index2]
        return result