def get_optimization_gradients(self):
        """Return the energy gradients of all geometries during an optimization"""
        grad_array = self.fields.get("Opt point       1 Gradient at each geome")
        if grad_array is None:
            return []
        else:
            return np.reshape(grad_array, (-1, len(self.molecule.numbers), 3))