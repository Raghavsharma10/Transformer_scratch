def _slopes_directions(self, data, dX, dY, method='tarboton'):
        """ Wrapper to pick between various algorithms
        """
        # %%
        if method == 'tarboton':
            return self._tarboton_slopes_directions(data, dX, dY)
        elif method == 'central':
            return self._central_slopes_directions(data, dX, dY)