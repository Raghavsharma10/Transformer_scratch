def get_coordinates(self, i, end=False):
        """Returns the starting coordinates of node i as a pair,
        or the end coordinates iff end is True."""
        if end:
            endpoint = self.endpoints[i][1]
        else:
            endpoint = self.endpoints[i][0]
        return (endpoint.real, endpoint.imag)