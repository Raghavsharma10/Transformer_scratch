def architecture(self):
        """Returns a dictionary describing the architecture of the layer."""
        arch = {'class': self.__class__,
                'n_in': self.n_in,
                'n_units': self.n_units,
                'activation_function': self.activation_function
                if hasattr(self, 'activation_function') else None}
        return arch